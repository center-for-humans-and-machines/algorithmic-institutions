"""Judge an LLM manager's PROMPT by replaying the real human games.

No rollout, no GPU, no contribution model: the state is real and identical
across prompt variants, so a difference between two variants is a difference
between the prompts rather than between two stacks. See
`src/aimanager/llm/replay.py` for what a decision point is.

Three tables per variant, written to --out:

  shape.csv      Mean punishment per contribution bin, on the evaluation
                 suite's own RPA bins, `contribution_valid` masked at source,
                 with the row count per bin. Beside them the TARGETING triple
                 from `scripts/rl_param_noise/targeting.py` -- a rank
                 correlation, the magnitude columns, and `profile_snr`, the
                 noise gate. Never the rank alone: a flat policy scores +-1 on
                 rank, and a rank over six bin means is attenuated by ties in
                 a way correlated with how quiet the manager is, so the tie
                 structure is reported next to it.
                 Reference rows: the human managers on the SAME decision
                 points, and the human managers on the whole deduped CSV
                 through `ResponseMetrics().rpa`, which is where the published
                 4.755 / 2.973 / 1.672 / 0.978 / 0.692 / 0.267 comes from.

  parse.csv      The parse failure rate and its breakdown by reason, the
                 answer format the model used, and the wasted-punishment rate
                 (punishment set for a player who gave no input, which the
                 environment discards). A run with a material failure rate is
                 not a result.

  agreement.csv  Per-decision comparison against what the real manager did on
                 the same state: mean absolute difference, exact-match rate,
                 and the correlation over agent-rounds. Diagnostic only --
                 the point is not to reproduce the human manager, and tuning
                 the prompt until it does would measure the tuning rather
                 than the model.

Usage, against a served endpoint:
    HOSTED_VLLM_API_BASE=http://localhost:8000/v1 \\
    python scripts/data_analysis/llm_prompt_replay.py \\
        --client endpoint:Qwen/Qwen3-8B --models Qwen3-8B \\
        --cache runs/{version}__{model}.jsonl \\
        --out plots/data_analysis/llm_prompt_replay/qwen3_8b

and to re-score completions already collected, with no model at all:
    python scripts/data_analysis/llm_prompt_replay.py \\
        --client cached --models Qwen3-8B Qwen3-32B \\
        --cache runs/{version}__{model}.jsonl --sample 30 \\
        --out plots/data_analysis/llm_prompt_replay/qwen3
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_EDGES,
    RPA_LABELS,
    ResponseMetrics,
)
from aimanager.llm.prompt import (  # noqa: E402
    DEFAULT_VERSION,
    PROMPT_VERSIONS,
)
from aimanager.llm.replay import (  # noqa: E402
    StubClient,
    decisions,
    human_frame,
    load_games,
    run_replay,
)

sys.path.insert(0, os.path.join(ROOT, "scripts"))

from rl_param_noise.targeting import targeting  # noqa: E402

# experiments/2group_8agent_50ep.csv through the evaluation suite's RPA bins.
HUMAN_PROFILE = [4.755, 2.973, 1.672, 0.978, 0.692, 0.267]


def bin_profile(frame, value="charged_punishment"):
    """Mean punishment per RPA contribution bin, masked at source.

    `contribution_valid` is applied here, not downstream: the simulation's
    per-round output stores the imputed median 9 on a timed-out player and the
    human CSV stores 0, and either would land in a bin the player never
    produced. `convert.load_sim` does not mask them, which is why this does.
    """
    valid = frame[frame["contribution_valid"]].dropna(subset=[value])
    bins = pd.cut(valid["contribution"], RPA_EDGES, labels=RPA_LABELS)
    grouped = valid.groupby(bins, observed=False)[value]
    return grouped.mean().reindex(RPA_LABELS), grouped.size().reindex(RPA_LABELS)


def profile_snr(frame, value="charged_punishment"):
    """The noise gate: range of the six bin means over their mean standard
    error. `rho` ranks six numbers; if they differ by less than their own
    sampling noise it ranks noise and returns +-1 anyway. `guard_report.py`
    takes the standard error across evaluation points; the only replication a
    replay carries is the game, so it is taken across episodes."""
    valid = frame[frame["contribution_valid"]].dropna(subset=[value]).copy()
    valid["bin"] = pd.cut(valid["contribution"], RPA_EDGES, labels=RPA_LABELS)
    per_episode = valid.pivot_table(
        index="episode_id", columns="bin", values=value, aggfunc="mean", observed=False
    )
    if len(per_episode) < 2:
        return float("nan")
    se = (per_episode.std() / np.sqrt(per_episode.count())).mean()
    spread = per_episode.mean().max() - per_episode.mean().min()
    if spread == 0:
        # a perfectly flat profile: no spread and no noise. Zero, not inf --
        # there is nothing here to rank, and the gate must say so.
        return 0.0
    if not se or not np.isfinite(se):
        return float("inf")
    return float(spread / se)


def tie_structure(means):
    """How many of the six bin means are distinct, and the smallest gap
    between neighbours. Ties are what attenuate a rank over six numbers, and a
    quiet manager ties more, so the rank is only readable beside this."""
    values = [m for m in means if np.isfinite(m)]
    rounded = np.round(values, 3)
    gaps = np.abs(np.diff(rounded))
    return {
        "distinct_bins": int(len(set(rounded.tolist()))),
        "min_neighbour_gap": float(gaps.min()) if len(gaps) else float("nan"),
    }


def markdown(table):
    """A markdown table without the optional `tabulate` dependency."""

    def cell(value):
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.3f}"
        return "" if value is None or pd.isna(value) else str(value)

    header = list(table.columns)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in header) + " |")
    return "\n".join(lines) + "\n"


def shape_row(frame, name, value="charged_punishment"):
    means, counts = bin_profile(frame, value)
    row = {"source": name}
    row.update({label: float(means[label]) for label in RPA_LABELS})
    row.update({f"n[{label}]": int(counts[label]) for label in RPA_LABELS})
    row.update(targeting(means.tolist(), counts.tolist()))
    row.update(tie_structure(means.tolist()))
    row["profile_snr"] = profile_snr(frame, value)
    return row


def parse_row(frame, name):
    """The failure rate, read from `parse.summarise` rather than recomputed.

    It is per ANSWER, not per agent-round: a failure is a failure of the whole
    answer, and counting it per player would weight it by group size.
    """
    row = {"source": name}
    row.update(frame.attrs["parse_summary"])
    row["agent_rounds"] = len(frame)
    no_input = ~frame["contribution_valid"]
    row["no_input_agent_rounds"] = int(no_input.sum())
    row["wasted_agent_rounds"] = int(frame["wasted"].sum())
    row["wasted_rate_on_no_input"] = (
        float(frame.loc[no_input, "wasted"].mean()) if no_input.any() else float("nan")
    )
    return row


def agreement_row(frame, name):
    both = frame.dropna(subset=["human_punishment"])
    both = both[both["contribution_valid"]]
    model = both["charged_punishment"].astype(float)
    human = both["human_punishment"].astype(float)
    return {
        "source": name,
        "agent_rounds": len(both),
        "mean_abs_diff": float((model - human).abs().mean()),
        "mean_signed_diff": float((model - human).mean()),
        "exact_match_rate": float((model == human).mean()),
        "pearson": float(model.corr(human)),
        "spearman": float(model.corr(human, method="spearman")),
        "mean_model": float(model.mean()),
        "mean_human": float(human.mean()),
    }


def human_reference_profile():
    """The published human profile, recomputed, as a check that this script's
    binning is the evaluation suite's binning and not a look-alike."""
    human = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    rpa = ResponseMetrics().rpa(human)
    means = rpa.groupby(level=0).mean().reindex(RPA_LABELS)
    counts = rpa.groupby(level=0).size().reindex(RPA_LABELS)
    row = {"source": "human managers (whole CSV, eval suite)"}
    row.update({label: float(means[label]) for label in RPA_LABELS})
    row.update({f"n[{label}]": int(counts[label]) for label in RPA_LABELS})
    row.update(targeting(means.tolist(), counts.tolist()))
    row.update(tie_structure(means.tolist()))
    row["profile_snr"] = float("nan")
    return row


# Four reference policies, run through the real prompt, the real stub client
# and the real parser. They are what says the battery discriminates before any
# model is asked: `human_table` should come back monotone decreasing with the
# human sign, `inverted` with the opposite sign, and the two flat ones should
# be caught by the noise gate and the tie structure rather than by `rho`,
# which has nothing to rank on them.
HUMAN_SHAPED = [5, 4, 3, 2, 2] + [1] * 8 + [0] * 8
STUB_POLICIES = {
    "human_table": lambda c: HUMAN_SHAPED[c],
    "never": lambda c: 0,
    "flat3": lambda c: 3,
    "inverted": lambda c: 0 if c <= 10 else 5,
}


def make_client(spec, rng):
    kind, _, name = spec.partition(":")
    if kind == "stub":
        table = STUB_POLICIES[name or "human_table"]

        def policy(decision):
            return {
                p.label: (0 if not p.gave_input else table(p.contribution))
                for p in decision.target.players
            }

        return StubClient(policy)
    if kind == "cached":

        def refuse(messages, decision=None, **_):
            raise KeyError(
                f"no cached completion for {decision.key}; "
                "run with --client endpoint or fill the cache first"
            )

        return refuse
    if kind == "endpoint":
        return _endpoint_client(name, rng)
    raise ValueError(f"unknown client {spec!r}")


def _endpoint_client(model, rng, api_base=None, temperature=0.0):
    """Any OpenAI-compatible server (vLLM via `vllm serve`, the MPCDF service).

    Deliberately plain urllib: the serving layer is a sibling's, this is only
    so the harness is runnable the moment an endpoint exists.
    """
    import json
    import urllib.request

    base = api_base or os.environ.get("HOSTED_VLLM_API_BASE")
    key = os.environ.get("HOSTED_VLLM_API_KEY", "local-no-auth")
    if not base:
        raise SystemExit("set HOSTED_VLLM_API_BASE or pass --api-base")

    def call(messages, decision=None, **_):
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 256,
        }
        if "qwen3" in model.lower():
            body["chat_template_kwargs"] = {"enable_thinking": False}
        request = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]

    return call


def paired_subset(points, args):
    """The decision points every cache answered, so the comparison stays paired.

    A collection can come back partial. Scoring each variant on whatever it
    happens to have would compare variants on different states, which is the
    one thing the replay design exists to avoid, so the intersection is taken
    instead and its size is printed.
    """
    answered = None
    for model in args.models:
        for version in args.variants:
            path = args.cache.replace("{version}", version).replace("{model}", model)
            if not os.path.exists(path):
                continue
            keys = set()
            with open(path) as handle:
                for line in handle:
                    if line.strip():
                        keys.add(json.loads(line)["key"].split("|", 1)[1])
            answered = keys if answered is None else (answered & keys)
    if answered is None:
        return points
    return [
        p for p in points if f"{p.episode_id}|{p.group_id}|{p.round_number}" in answered
    ]


def sample_points(points, n, seed):
    """A fixed subsample, identical across variants so the comparison is
    paired: the same states, the same group sizes, the same bins."""
    if n is None or n >= len(points):
        return points
    rng = np.random.default_rng(seed)
    index = rng.choice(len(points), size=n, replace=False)
    return [points[i] for i in sorted(index)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=sorted(PROMPT_VERSIONS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=[""],
        help=(
            "labels for the models whose completions to score, e.g. Qwen3-8B "
            "Qwen3-32B. Every (variant, model) pair whose cache file exists is "
            "scored into one table; the rest are reported as missing."
        ),
    )
    parser.add_argument("--client", default="stub:human_table")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--data", default=os.path.join(ROOT, HUMAN_DATA_FILE))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dump-prompts",
        default=None,
        help="write one JSON per prompt instead of calling a model",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help=(
            "score the four reference policies instead of prompt variants, to "
            "show what the battery separates before any model is asked"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    points = sample_points(decisions(load_games(args.data)), args.sample, args.seed)
    print(f"{len(points)} decision points")

    if args.dump_prompts:
        _dump(points, args.variants, args.dump_prompts)
        return

    rng = np.random.default_rng(args.seed)
    if args.client == "cached" and args.cache:
        points = paired_subset(points, args)
        print(f"{len(points)} answered by every cache -- the paired subset")

    frames = {}
    if args.reference:
        for policy in STUB_POLICIES:
            frames[policy] = run_replay(
                points, make_client(f"stub:{policy}", rng), version=DEFAULT_VERSION
            )
    for model in args.models if not args.reference else []:
        for version in args.variants:
            cache = args.cache
            if cache:
                cache = cache.replace("{version}", version).replace("{model}", model)
                if args.client == "cached" and not os.path.exists(cache):
                    print(f"missing: {version} [{model}] -- no {cache}")
                    continue
            client = make_client(args.client, rng)
            if args.client.startswith("endpoint") and args.api_base:
                client = _endpoint_client(
                    args.client.partition(":")[2], rng, args.api_base, args.temperature
                )
            label = f"{version} [{model}]" if model else version
            frames[label] = run_replay(
                points, client, version=version, cache_path=cache
            )
            slug = label.replace(" ", "_").replace("[", "").replace("]", "")
            frames[label].to_csv(
                os.path.join(args.out, f"per_round_{slug}.csv"), index=False
            )

    reference = human_frame(points)
    shape = pd.DataFrame(
        [
            human_reference_profile(),
            shape_row(reference, "human managers (same decision points)"),
        ]
        + [shape_row(f, v) for v, f in frames.items()]
    )
    parse = pd.DataFrame([parse_row(f, v) for v, f in frames.items()])
    agreement = pd.DataFrame([agreement_row(f, v) for v, f in frames.items()])

    for name, table in (
        ("shape", shape),
        ("parse", parse),
        ("agreement", agreement),
    ):
        table.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        with open(os.path.join(args.out, f"{name}.md"), "w") as handle:
            handle.write(markdown(table))
        print(f"\n== {name}\n{table.to_string(index=False, float_format='%.3f')}")

    drift = np.abs(np.array(shape.loc[0, RPA_LABELS], dtype=float) - HUMAN_PROFILE)
    print(f"\nhuman reference reproduced to {drift.max():.4f}")


def _dump(points, variants, out_dir):
    from aimanager.llm.prompt import build_prompt

    os.makedirs(out_dir, exist_ok=True)
    for version in variants:
        path = os.path.join(out_dir, f"prompts_{version}.jsonl")
        with open(path, "w") as handle:
            for point in points:
                prompt = build_prompt(point.records, version=version)
                handle.write(
                    json.dumps(
                        {
                            "key": f"{version}|{point.episode_id}"
                            f"|{point.group_id}|{point.round_number}",
                            "system": prompt.system,
                            "user": prompt.user,
                            "labels": list(prompt.labels),
                            "fingerprint": prompt.fingerprint,
                        }
                    )
                    + "\n"
                )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
