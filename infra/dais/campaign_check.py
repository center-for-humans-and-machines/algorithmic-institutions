"""Did a campaign (or one instance's version) reach its expected state?

    python infra/dais/campaign_check.py campaign <campaign.yml> [--expect scored|finished]
    python infra/dais/campaign_check.py resume <instance> [--expect ...]

campaign: every generation the campaign file asks for (1 + generations) has
started and finished, and each of its versions is `scored` (--expect
finished: `scored` or `failed`). resume: the instance's current version is.
Prints one line per version (status, score, agent time and tokens, network
audit findings) and exits 1 when the expectation is not met. Reads PF_RUNS.
"""

import argparse
import json
import re
import statistics
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "policy_finder"))


#: Steps whose failure is the agent's work (its runs, what it wrote, its
#: rule): a failed version there counts; a failure at any other step (sims,
#: fetches, the fit) is the infrastructure's, and --expect finished refuses it
AGENT_STEPS = ("agent", "collect", "check", "image_check")


def failure(state):
    """(failed step, reason, the agent's?) of a failed version."""
    import campaign

    step = campaign.Campaign._failed_step(state)
    reason = (state.get("error") or "").strip().splitlines()
    reason = reason[0][:300] if reason else ""
    data = state.get("data") or {}
    own = step in AGENT_STEPS
    error = state.get("error") or ""
    if step == "agent" and not data.get("agent_runs"):
        own = False  # no attempt of the agent's: its runs were lost
    if step == "check" and re.search(r"FAIL: cannot (read|diff|list)", error):
        own = False  # check_instance.sh could not read the worktree
    if step == "agent":
        # an agent run that failed on the endpoint is not the agent's
        import agent_apptainer

        for r in data.get("agent_runs") or []:
            errors = agent_apptainer.provider_errors(r.get("result") or {})
            if agent_apptainer.ENDPOINT_ERRORS.search(errors):
                own = False
    return step, reason, own


def describe(stem, state):
    data = state.get("data") or {}
    score = (data.get("score") or {}).get("pool")
    runs = data.get("agent_runs") or []
    agent = []
    for r in runs:
        res = r.get("result") or {}
        audit = r.get("network_audit") or {}
        agent.append(f"{r.get('run_id')} {r.get('status')} {res.get('seconds')} s"
                     f" tokens {res.get('tokens')}"
                     f" compactions {res.get('compactions')}"
                     f" overflow {res.get('overflow') or 'no'}"
                     f" audit {len(audit.get('findings', []))}")
    return (f"{stem}: {state.get('status')}"
            + (f" pool {score:.2f}" if isinstance(score, (int, float)) else "")
            + (f"; agent runs: {'; '.join(agent)}" if agent else "")
            + (f"; lost runs {len(data['lost_runs'])}" if data.get("lost_runs") else "")
            + (f"; step seconds {state['times']}" if state.get("times") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("campaign", "resume"))
    ap.add_argument("target")
    ap.add_argument("--expect", choices=("scored", "finished"), default="scored")
    ap.add_argument("--min-first-scored", type=int,
                    help="at least this many versions of the first agent generation"
                    " scored (single-idea mode: the second, after the seed)")
    ap.add_argument("--no-expired", action="store_true",
                    help="no agent run may have expired (its time limit), and none "
                    "may have failed on the endpoint (the load test)")
    ap.add_argument("--no-limit-stops", action="store_true",
                    help="no agent run may have been stopped at the time limit or "
                    "had a step cut at the output limit (stage option +nolimits)")
    ap.add_argument("--json-out", help="write the limit report here as JSON")
    args = ap.parse_args()
    ok_status = ("scored",) if args.expect == "scored" else ("scored", "failed")
    pf_runs = Path(os.environ["PF_RUNS"])
    wt_root = Path(os.environ.get("PF_WORKTREE_ROOT") or pf_runs / "worktrees")
    bad = []
    if args.mode == "campaign":
        import campaign

        c = campaign.Campaign(args.target, pf_runs)
        gens = c.state.get("generations", [])
        want = 1 + c.settings["generations"]
        if len(gens) < want or not all(g.get("finished") for g in gens[:want]):
            bad.append(f"{sum(bool(g.get('finished')) for g in gens)} of {want}"
                       " generations finished")
        members = [(m["name"], m["version"]) for g in gens for m in g["members"]]
        # the first generation of agent versions: in single-idea mode the
        # first one is the seed, written without an agent
        lead = 1 if c.settings.get("mode") == "single_idea" else 0
        first = {(m["name"], m["version"])
                 for m in (gens[lead]["members"] if len(gens) > lead else [])}
    else:
        conf = json.loads((wt_root / args.target / ".claude/policy_finder.json").read_text())
        members = [(args.target, conf.get("version", 1))]
        first = set(members)
    seconds, first_scored, expired, endpoint, lost, rejected = [], 0, [], [], 0, []
    time_stops, output_stops, lost_to_limits, limits_seen = [], [], [], set()
    loop_stops = []
    overflows, overflows_recovered, limit_hit, length_cut = 0, 0, 0, 0
    for name, version in members:
        stem = name if version == 1 else f"{name}-v{version}"
        path = wt_root / f"{stem}.state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        print(describe(stem, state))
        data = state.get("data") or {}
        endpoint += [f"{stem} {r.get('run_id')}" for r in data.get("endpoint_runs") or []]
        lost += len(data.get("lost_runs") or [])
        # single-idea mode: agent runs whose notes the check rejected (each a
        # failed attempt, retried; pipeline.step_agent)
        rejected += [f"{stem} {r.get('run_id')}: {'; '.join(r['note_rejected'])[:200]}"
                     for r in data.get("agent_runs") or [] if r.get("note_rejected")]
        stopped_here = []
        for r in data.get("agent_runs") or []:
            if (r.get("result") or {}).get("seconds") is not None:
                seconds.append(r["result"]["seconds"])
            if r.get("time_limit_s"):
                limits_seen.add((r["time_limit_s"], r.get("opencode_time_limit_s")))
            ran = f"{stem} {r.get('run_id')}"
            if r.get("status") == "expired" or (r.get("result") or {}).get("timed_out"):
                time_stops.append(ran)
                stopped_here.append("time limit")
            if (r.get("result") or {}).get("length_stops"):
                output_stops.append(ran)
                stopped_here.append("output limit")
            # stopped in a loop of failing tool calls (opencode_run LoopWatch):
            # a failed attempt of the model's, not a limit
            if (r.get("result") or {}).get("tool_loop"):
                loop = r["result"]["tool_loop"]
                loop_stops.append(f"{ran} ({loop.get('repeats')}x {loop.get('tool')})")
            if r.get("status") == "expired":
                expired.append(f"{stem} {r.get('run_id')}")
            res = r.get("result") or {}
            # attempts the server refused past its context (failed attempts,
            # also when OpenCode recovered by compacting)
            if res.get("overflow"):
                overflows += 1
                overflows_recovered += bool(res["overflow"].get("recovered"))
            # OpenCode stopped at the runner's time limit (--timeout)
            limit_hit += bool(res.get("timed_out"))
            # a step that ran out of output tokens (failed attempts)
            length_cut += bool(res.get("length_stops"))
        if (state.get("status") == "failed" and failure(state)[0] == "agent"
                and stopped_here):
            # all its attempts failed, one of them at a limit: rerun it with a
            # higher cap rather than count it as the model's failure
            lost_to_limits.append(f"{stem} ({', '.join(sorted(set(stopped_here)))})")
        if (name, version) in first and state.get("status") == "scored":
            first_scored += 1
        if state.get("status") == "failed":
            step, reason, own = failure(state)
            print(f"  failed at {step} ({'the agent' if own else 'INFRASTRUCTURE'}):"
                  f" {reason}")
            if not own:
                bad.append(f"{stem} failed at {step} for an infrastructure reason")
                continue
        if state.get("status") not in ok_status:
            bad.append(f"{stem} is {state.get('status')}")
    durations = {}
    if seconds:
        q = statistics.quantiles(seconds, n=4) if len(seconds) > 1 else seconds * 3
        durations = {"n": len(seconds), "min": min(seconds),
                     "quartiles": [round(x) for x in q], "max": max(seconds)}
        print(f"agent run seconds: n {len(seconds)}, min {min(seconds)}, quartiles "
              f"{durations['quartiles']}, max {max(seconds)}")
        for limit, oc_limit in sorted(limits_seen):
            # OpenCode runs to oc_limit; the container is killed at limit
            share = max(seconds) / oc_limit if oc_limit else None
            used = f"; the longest run used {share:.0%} of OpenCode's" if share else ""
            print(f"  agent time limit {limit} s (OpenCode stopped at {oc_limit} s, "
                  f"600 s before it){used}")
            durations.setdefault("limits", []).append(
                {"time_limit_s": limit, "opencode_time_limit_s": oc_limit,
                 "longest_share_of_opencode_limit": share and round(share, 3)})
    print(f"runs stopped at the time limit: {len(time_stops)} {time_stops}")
    print(f"runs with a step cut at the output limit: {len(output_stops)} {output_stops}")
    print(f"runs stopped in a loop of failing tool calls: {len(loop_stops)} {loop_stops}")
    print(f"versions lost with a limit stop among their attempts (rerun them with a "
          f"higher cap): {len(lost_to_limits)} {lost_to_limits}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"time_limit_stops": time_stops, "output_limit_stops": output_stops,
             "versions_lost_with_limit_stops": lost_to_limits,
             "tool_loop_stops": loop_stops,
             "agent_run_seconds": durations}, indent=1) + "\n")
    print(f"first generation: {first_scored} of {len(first)} scored; expired runs: "
          f"{len(expired)}; endpoint runs: {len(endpoint)}; lost runs: {lost}; "
          f"overflow attempts (failed): {overflows}, of which OpenCode recovered: "
          f"{overflows_recovered}; runs stopped at the runner's time limit: {limit_hit}; "
          f"runs with a step cut at the output limit (failed): {length_cut}")
    print(f"notes rejected by the check (failed attempts, retried): {len(rejected)}")
    for line in rejected[:20]:
        print(f"  {line}")
    if args.min_first_scored is not None and first_scored < args.min_first_scored:
        bad.append(f"{first_scored} first-generation versions scored, fewer than "
                   f"{args.min_first_scored}")
    if args.no_expired and expired:
        bad.append(f"expired agent runs: {', '.join(expired)}")
    if args.no_expired and endpoint:
        bad.append(f"agent runs failed on the endpoint: {', '.join(endpoint)}")
    if args.no_limit_stops and (time_stops or output_stops):
        bad.append(f"agent runs stopped at a limit: time {time_stops}, "
                   f"output {output_stops}")
    if bad:
        print("NOT AS EXPECTED: " + "; ".join(bad))
        return 1
    print(f"as expected: {len(members)} version(s) {args.expect}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
