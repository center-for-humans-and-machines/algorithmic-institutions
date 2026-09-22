"""Measure what the LLM manager costs, before anything depends on it.

A rollout is `n_rounds` sequential batched calls, one prompt per episode per
round, so wall clock is set by how fast a server clears a batch of B and by how
long the trace has grown by the last round. Two numbers decide how the budget
should be spent:

* **the saturation width** -- the batch width at which a server stops going
  faster. Below it, episodes are free: a wider batch costs the same wall clock.
  Above it, episodes cost linearly. This is the episode count to aim for.
* **completions per second per card**, which says what another GPU buys.

Two modes, both run by default:

`--saturation` fires one batch of each width at a realistic prompt (a trace
already `--saturation-round` rounds deep) and reports throughput against width.
It does not run a rollout.

`--episodes` runs full `n_rounds` rollouts at each episode count, driving
`LLMManager.predict` with state dicts shaped exactly as
`ArtificialHumanEnv.served_state()` produces them, with the real accumulating
trace. It deliberately does NOT run the environment: the artificial-human
models are CPU-bound and cheap next to the language model, and leaving them out
measures the part that is at risk without needing their artifacts on the node.

    python scripts/llm_manager/throughput_benchmark.py \
        --episodes 200 1000 3000 --rounds 24 --out bench.json

The endpoints come from `HOSTED_VLLM_API_BASE`, which
`scripts/llm_manager/serve_vllm.slurm.sh` exports as a comma-separated list.
"""

import argparse
import json
import os
import sys
import time

import torch as th

from aimanager.manager.llm_manager import (
    DEFAULT_OBJECTIVE,
    LLMManager,
    PlayerRound,
    RoundContext,
    RoundRecord,
)

N_AGENTS = 8
AGENT_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
MAX_CONTRIBUTION = 20


def synthetic_state(n_batch, round_number, generator, prev=None):
    """A `served_state()`-shaped dict: every key the manager reads, (B, A, 1).

    Contributions are drawn rather than simulated: the manager's cost depends
    on the size of the numbers in the trace, not on where they came from.
    """
    size = (n_batch, N_AGENTS, 1)
    contribution = th.randint(
        0, MAX_CONTRIBUTION + 1, size, generator=generator, dtype=th.int64
    )
    valid = th.rand(size, generator=generator) > 0.05
    state = {
        "contribution": contribution,
        "contribution_valid": valid,
        "punishment": th.zeros(size, dtype=th.int64),
        "punishment_valid": th.ones(size, dtype=th.bool),
        "round_number": th.full(size, round_number, dtype=th.int64),
        "is_first": th.full(size, round_number == 0, dtype=th.bool),
        "agent_group": th.tensor(AGENT_GROUPS, dtype=th.int64)
        .reshape(1, N_AGENTS, 1)
        .expand(size)
        .contiguous(),
        "common_good": th.zeros(size, dtype=th.float),
        "contributor_payoff": th.zeros(size, dtype=th.float),
    }
    if prev is None:
        state["prev_contribution"] = th.zeros(size, dtype=th.int64)
        state["prev_punishment"] = th.zeros(size, dtype=th.int64)
        state["prev_contribution_valid"] = th.zeros(size, dtype=th.bool)
        state["prev_common_good"] = th.zeros(size, dtype=th.float)
    else:
        state["prev_contribution"] = prev["contribution"]
        state["prev_punishment"] = prev["punishment_applied"]
        state["prev_contribution_valid"] = prev["contribution_valid"]
        # 1.6 * c - p, the pool per member.
        state["prev_common_good"] = 1.6 * prev["contribution"].to(th.float) - prev[
            "punishment_applied"
        ].to(th.float)
    return state


def one_rollout(manager, n_batch, n_rounds, seed):
    """Run `n_rounds` batched calls and time every one."""
    generator = th.Generator().manual_seed(seed)
    manager.reset()
    stats = manager.client.stats
    prev = None
    per_round = []
    t_start = time.time()
    for round_number in range(n_rounds):
        state = synthetic_state(n_batch, round_number, generator, prev)
        before = (stats.prompt_tokens, stats.completion_tokens)
        t0 = time.time()
        punishment, _ = manager.predict(state)
        dt = time.time() - t0
        per_round.append(
            {
                "round": round_number,
                "wall_s": round(dt, 3),
                "prompt_tokens": stats.prompt_tokens - before[0],
                "completion_tokens": stats.completion_tokens - before[1],
            }
        )
        # Feed back what the game would have charged: a punishment aimed at a
        # player who gave no input is zeroed (ArtificialHumanEnv.punish).
        charged = th.where(
            state["contribution_valid"], punishment, th.zeros_like(punishment)
        )
        prev = {
            "contribution": state["contribution"],
            "contribution_valid": state["contribution_valid"],
            "punishment_applied": charged,
        }
    return time.time() - t_start, per_round


def deep_context(strategy, history_rounds, n_players, n_punishments, objective):
    """A context whose trace is already `history_rounds` deep."""
    history = tuple(
        RoundRecord(
            round_number=t,
            players=tuple(
                PlayerRound(
                    agent=a,
                    contribution=(3 * a + t) % (MAX_CONTRIBUTION + 1),
                    punishment=(a + t) % 8,
                    contribution_valid=(a + t) % 17 != 0,
                )
                for a in range(n_players)
            ),
            common_good=12.0 + t,
        )
        for t in range(history_rounds)
    )
    return RoundContext(
        episode=0,
        round_number=history_rounds,
        n_players=n_players,
        n_punishments=n_punishments,
        objective=objective,
        prompt_version=getattr(strategy, "version", "v0"),
        history=history,
        current=tuple(
            PlayerRound(
                agent=a,
                contribution=(5 * a) % (MAX_CONTRIBUTION + 1),
                punishment=0,
                contribution_valid=True,
            )
            for a in range(n_players)
        ),
        n_rounds=24,
        switch_every=4,
    )


def saturation_sweep(manager, widths, history_rounds, n_players):
    """One batch at each width, at a realistic prompt length."""
    strategy = manager.strategy
    ctx = deep_context(
        strategy,
        history_rounds,
        n_players,
        manager.n_punishments,
        manager.objective,
    )
    messages = strategy.build(ctx)
    constraint = manager._constraint_for(ctx)  # noqa: SLF001 - same package
    rows = []
    for width in widths:
        stats = manager.client.stats
        before = (stats.n_calls, stats.prompt_tokens, stats.completion_tokens)
        t0 = time.time()
        results = manager.client.complete(
            [messages] * width,
            meta=[{"episode": i, "round": history_rounds} for i in range(width)],
            constraints=[constraint] * width,
        )
        dt = time.time() - t0
        n_ok = sum(1 for r in results if r.ok)
        rows.append(
            {
                "width": width,
                "wall_s": round(dt, 3),
                "completions_per_s": round(width / dt, 2) if dt else 0.0,
                "ok": n_ok,
                "errors": width - n_ok,
                "mean_prompt_tokens": round(
                    (stats.prompt_tokens - before[1]) / max(width, 1), 1
                ),
                "mean_completion_tokens": round(
                    (stats.completion_tokens - before[2]) / max(width, 1), 1
                ),
            }
        )
        print(
            f"[bench] width {width:>5}: {dt:7.2f}s  "
            f"{rows[-1]['completions_per_s']:8.2f} completions/s  "
            f"errors {rows[-1]['errors']}"
        )
    return rows


def saturation_width(rows, tolerance=0.10):
    """The smallest width within `tolerance` of the best throughput seen.

    Below it, adding episodes is nearly free; above it, cost is linear.
    """
    if not rows:
        return None
    best = max(r["completions_per_s"] for r in rows)
    for row in rows:
        if row["completions_per_s"] >= best * (1.0 - tolerance):
            return row["width"]
    return rows[-1]["width"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="*", default=[200, 1000, 3000])
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument(
        "--saturation",
        type=int,
        nargs="*",
        default=[16, 64, 128, 256, 512, 1024, 2048],
        help="batch widths for the saturation sweep; empty to skip",
    )
    parser.add_argument("--saturation-round", type=int, default=12)
    parser.add_argument("--n-players", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("LLM_MANAGER_MODEL"))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-concurrent", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="v0")
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=int(os.environ.get("LLM_MANAGER_N_GPUS", 0)) or None,
        help="cards behind the endpoints, for per-card throughput",
    )
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        nargs="+",
        default=[300, 1000, 6144],
        help="episode counts the existing baselines use, for extrapolation",
    )
    args = parser.parse_args(argv)

    if not args.model:
        parser.error("--model, or LLM_MANAGER_MODEL, is required")

    def build_manager(log_name=None):
        log_path = None
        if args.log_dir and log_name:
            log_path = os.path.join(args.log_dir, log_name)
        return LLMManager(
            model=args.model,
            api_base=args.api_base,
            prompt_version=args.prompt_version,
            objective=DEFAULT_OBJECTIVE,
            group_id=args.group_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
            n_rounds=args.rounds,
            switch_every=4,
            log_path=log_path,
        )

    probe = build_manager()
    results = {
        "model": args.model,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_concurrent": args.max_concurrent,
        "prompt_version": args.prompt_version,
        "n_endpoints": probe.client.n_endpoints,
        "n_gpus": args.n_gpus,
        "endpoints": probe.client.endpoints,
        "constrained_decode": probe.constrained_decode,
        "saturation": [],
        "rollouts": [],
    }

    if args.saturation:
        print(
            f"[bench] saturation sweep at a {args.saturation_round}-round trace, "
            f"{probe.client.n_endpoints} endpoint(s)"
        )
        rows = saturation_sweep(
            probe, args.saturation, args.saturation_round, args.n_players
        )
        results["saturation"] = rows
        width = saturation_width(rows)
        peak = max((r["completions_per_s"] for r in rows), default=0.0)
        results["saturation_width"] = width
        results["peak_completions_per_s"] = peak
        if args.n_gpus:
            results["peak_completions_per_s_per_card"] = round(peak / args.n_gpus, 2)
        print(
            f"[bench] saturates at width ~{width}; peak {peak:.1f} completions/s"
            + (f" ({peak / args.n_gpus:.1f} per card)" if args.n_gpus else "")
        )

    for n_batch in args.episodes:
        manager = build_manager(f"calls_b{n_batch}.jsonl")
        print(f"[bench] rollout: {n_batch} episodes x {args.rounds} rounds")
        wall, per_round = one_rollout(manager, n_batch, args.rounds, args.seed)
        stats = manager.client.stats
        n_calls = max(stats.n_calls, 1)
        entry = {
            "episodes": n_batch,
            "rollout_wall_s": round(wall, 2),
            "rollout_minutes": round(wall / 60.0, 2),
            "n_calls": stats.n_calls,
            "completions_per_s": round(stats.n_calls / wall, 2) if wall else 0.0,
            "mean_prompt_tokens": round(stats.prompt_tokens / n_calls, 1),
            "mean_completion_tokens": round(stats.completion_tokens / n_calls, 1),
            "total_prompt_tokens": stats.prompt_tokens,
            "total_completion_tokens": stats.completion_tokens,
            "truncated": stats.truncated,
            "parse_failures": manager.parse_failures,
            "parse_failure_rate": round(manager.parse_failure_rate, 5),
            "call_errors": manager.call_errors,
            "first_round_wall_s": per_round[0]["wall_s"],
            "last_round_wall_s": per_round[-1]["wall_s"],
            "first_round_prompt_tokens": per_round[0]["prompt_tokens"],
            "last_round_prompt_tokens": per_round[-1]["prompt_tokens"],
            "per_round": per_round,
            "report": manager.report(),
        }
        if args.n_gpus:
            entry["completions_per_s_per_card"] = round(
                stats.n_calls / wall / args.n_gpus, 2
            )
        # Linear in episodes only ABOVE saturation. Marked as extrapolated so
        # it is never mistaken for a measurement.
        per_episode = wall / n_batch if n_batch else 0.0
        entry["extrapolated_hours_if_linear"] = {
            str(n): round(per_episode * n / 3600.0, 2) for n in args.baseline_episodes
        }
        results["rollouts"].append(entry)
        print(
            f"[bench] {n_batch} episodes: {wall:.1f}s "
            f"({wall / 60:.1f} min, {stats.n_calls / wall:.1f} completions/s), "
            f"prompt {entry['mean_prompt_tokens']:.0f} tok "
            f"(round 0 {per_round[0]['prompt_tokens'] / n_batch:.0f} -> "
            f"round {args.rounds - 1} "
            f"{per_round[-1]['prompt_tokens'] / n_batch:.0f}), "
            f"completion {entry['mean_completion_tokens']:.1f} tok, "
            f"parse failures {manager.parse_failures}"
        )

    text = json.dumps(results, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"[bench] wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
