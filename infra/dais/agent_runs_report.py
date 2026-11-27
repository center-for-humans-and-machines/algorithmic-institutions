"""One table row per agent run of a campaign: what the results report needs.

    python infra/dais/agent_runs_report.py <campaign.yml> [<campaign.yml> ...]

Per version (from its state file in PF_RUNS) and per agent run: the version's
status and score, the check's verdict, the run's status, error and failures,
seconds, tokens (input, output, cache read), models, how the session ended
(the final assistant message's finish reason and whether it has text), the
report's length, whether the session compacted (a summary message), and the
network audit's findings. Reads the transcripts the pipeline copied beside
the state file (<stem>.transcripts/<run>/session.json). Writes TSV to stdout.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/policy_finder"))
import campaign  # noqa: E402

COLUMNS = ("campaign", "generation", "stem", "status", "pool", "failed_step", "check",
           "run", "run_status", "seconds", "input", "output", "cache_read", "models",
           "final_finish", "final_has_text", "report_chars", "compacted", "messages",
           "audit_findings", "error")


def session_facts(path):
    try:
        session = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    msgs = session.get("messages") or []
    assistant = [m for m in msgs if (m.get("info") or {}).get("role") == "assistant"]
    compacted = any((m.get("info") or {}).get("summary")
                    or any(p.get("type") == "compaction" for p in m.get("parts") or [])
                    for m in msgs)
    final = assistant[-1] if assistant else {}
    text = "".join(p.get("text", "") for p in final.get("parts") or []
                   if p.get("type") == "text")
    return {"final_finish": (final.get("info") or {}).get("finish"),
            "final_has_text": bool(text.strip()), "compacted": compacted,
            "messages": len(msgs)}


def rows(path, pf_runs, wt_root):
    c = campaign.Campaign(path, pf_runs)
    for g in c.state.get("generations", []):
        for m in g["members"]:
            stem = campaign.stem_of(m["name"], m["version"])
            state_path = wt_root / f"{stem}.state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            data = state.get("data") or {}
            score = (data.get("score") or {}).get("pool")
            check = data.get("check") or ""
            base = {"campaign": c.path.stem, "generation": g["index"], "stem": stem,
                    "status": state.get("status"),
                    "pool": "" if score is None else f"{score:.2f}",
                    "failed_step": (campaign.Campaign._failed_step(state)
                                    if state.get("status") == "failed" else ""),
                    "check": "ok" if "OK:" in check else ("" if not check else "fail")}
            runs = (data.get("lost_runs") or []) + (data.get("agent_runs") or [])
            if not runs:
                yield base
            for r in runs:
                res = r.get("result") or {}
                tokens = res.get("tokens") or {}
                session = wt_root / f"{stem}.transcripts" / r["run_id"] / "session.json"
                yield {**base, "run": r["run_id"], "run_status": r.get("status"),
                       "seconds": res.get("seconds"), "input": tokens.get("input"),
                       "output": tokens.get("output"), "cache_read": tokens.get("cache_read"),
                       "models": ",".join(r.get("models") or []),
                       "report_chars": len(r.get("report") or "")
                       or (len(Path(wt_root / f"{stem}.report.md").read_text())
                           if (wt_root / f"{stem}.report.md").exists()
                           and r.get("status") == "succeeded" else 0),
                       "audit_findings": len((r.get("network_audit") or {})
                                             .get("findings") or []),
                       "error": (r.get("error") or "")[:200].replace("\t", " ")
                       .replace("\n", " "),
                       **session_facts(session)}


def main():
    pf_runs = Path(os.environ["PF_RUNS"])
    wt_root = Path(os.environ.get("PF_WORKTREE_ROOT") or pf_runs / "worktrees")
    print("\t".join(COLUMNS))
    for path in sys.argv[1:]:
        for row in rows(path, pf_runs, wt_root):
            print("\t".join("" if row.get(k) is None else str(row.get(k, "")) for k in COLUMNS))


if __name__ == "__main__":
    main()
