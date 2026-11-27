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
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "policy_finder"))


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
    else:
        conf = json.loads((wt_root / args.target / ".claude/policy_finder.json").read_text())
        members = [(args.target, conf.get("version", 1))]
    for name, version in members:
        stem = name if version == 1 else f"{name}-v{version}"
        path = wt_root / f"{stem}.state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        print(describe(stem, state))
        if state.get("status") not in ok_status:
            bad.append(f"{stem} is {state.get('status')}")
    if bad:
        print("NOT AS EXPECTED: " + "; ".join(bad))
        return 1
    print(f"as expected: {len(members)} version(s) {args.expect}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
