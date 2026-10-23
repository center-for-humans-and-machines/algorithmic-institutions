"""Bookkeeping for the observation app's Summary (the policy-finder-summarizer skill).

The Summary's categories come from Claude reading the rule configs. They are
kept in summary.json beside the instances (outside every worktree, so no agent
reads them):

    {"categories": {"<name>": {"kind": "rule" | "policy category",
                               "meaning": "..."}},
     "instances": {"<version>": {"rule_sha": "...", "summarized": "<date>",
                                 "rules_included": ["<rule>", ...],
                                 "policy_category": "<policy category>",
                                 "idea": "<one line>"}}}

A rule is one part a policy is built from (who it punishes, when it holds
back, what it caps or forgives); a policy category is a policy's overall
strategy, one per version, a few in all.

Each version of an instance is recorded on its own (#244), under its
rule_name: `<name>` for version 1, `<name>-v<k>` for version k.

The numbers (pool, se, members) and the inputs a rule reads are not stored:
the app reads them from each sweep and rule on every load.

summary.json and references.json (the Summary's reference pools) are kept on
Raven, in OBSERVATORY (/u/certuer/policy-finder-observatory, a fixed folder in
one account, so everyone means the same place; reached through the `raven`
ssh alias): the one copy that is kept and updated. The files beside the instances are
working copies, which the app and the export read. `status` pulls them first;
`record` and `category` pull, write and push back. Without the SSH
ControlMaster (`ssh raven`) they warn and work on the local copies; `push`
sends them once Raven is reachable again. None of this is on
policy-finder-base: every agent starts from that branch and can read all of
it, and these files describe earlier rules and their results.

Usage:
    python scripts/policy_finder/observation_app/summarize.py status
    python scripts/policy_finder/observation_app/summarize.py record <version> \
        --rule <rule> [--rule <rule> ...] --policy-category <category> \
        --idea "<one line>"
    python scripts/policy_finder/observation_app/summarize.py category <name> \
        --kind rule|policy-category --meaning "<one line>"
    python scripts/policy_finder/observation_app/summarize.py pull | push

`status` lists the versions to summarize: a rule not recorded yet, or changed
since (by its hash). `record` stores one instance against its current rule;
categories must exist first (`category`). PF_WORKTREE_ROOT as in
new_instance.sh.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

MAIN = Path(__file__).resolve().parents[3]
WT_ROOT = Path(os.environ.get("PF_WORKTREE_ROOT", MAIN.parent / "policy-finder-worktrees"))
SUMMARY = WT_ROOT / "summary.json"
#: Where summary.json and references.json are kept (see the module docstring)
OBSERVATORY = os.environ.get(
    "PF_OBSERVATORY", "raven:/u/certuer/policy-finder-observatory"
)
SYNCED = ("summary.json", "references.json")
#: The category kinds, as the command line names them and as stored
KINDS = {"rule": "rule", "policy-category": "policy category"}
RULE = "configs/managers/rule_based/{name}.yml"
SWEEP = "plots/simulation/policy_finder/{name}_sweep/sweep.json"
#: A later version's name, <instance>-v<k> (#244)
LATER = re.compile(r"^(?P<instance>.+)-v(?P<k>[0-9]+)$")


def _raven_up() -> bool:
    host = OBSERVATORY.split(":", 1)[0]
    ok = subprocess.run(["ssh", "-O", "check", host], capture_output=True).returncode == 0
    if not ok:
        print(
            f"warning: no SSH ControlMaster to {host} (run `ssh {host}`):"
            f" working on the local copies in {WT_ROOT}",
            file=sys.stderr,
        )
    return ok


def pull():
    """Copy the kept files from OBSERVATORY over the local working copies."""
    if not _raven_up():
        return
    for name in SYNCED:
        # a file not on Raven yet keeps its local copy
        subprocess.run(
            ["rsync", "-a", f"{OBSERVATORY}/{name}", str(WT_ROOT / name)],
            capture_output=True,
        )


def push():
    """Copy the local working copies to OBSERVATORY."""
    if not _raven_up():
        return
    host, folder = OBSERVATORY.split(":", 1)
    subprocess.run(["ssh", host, f"mkdir -p {folder}"], check=True)
    files = [str(WT_ROOT / n) for n in SYNCED if (WT_ROOT / n).exists()]
    subprocess.run(["rsync", "-a", *files, f"{OBSERVATORY}/"], check=True)
    print(f"pushed {', '.join(Path(f).name for f in files)} to {OBSERVATORY}")


def load() -> dict:
    try:
        data = json.loads(SUMMARY.read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("categories", {})
    data.setdefault("instances", {})
    return data


def save(data: dict):
    SUMMARY.write_text(json.dumps(data, indent=2) + "\n")


def instance_of(version: str) -> str:
    """The instance a version belongs to: its worktree's name."""
    m = LATER.match(version)
    if m and (WT_ROOT / m["instance"] / ".claude/policy_finder.json").exists():
        return m["instance"]
    return version


def rule_path(version: str) -> Path:
    return WT_ROOT / instance_of(version) / RULE.format(name=version)


def sweep_path(version: str) -> Path:
    return WT_ROOT / instance_of(version) / SWEEP.format(name=version)


def rule_sha(version: str):
    path = rule_path(version)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def instances() -> list:
    return sorted(
        d.name for d in WT_ROOT.iterdir() if (d / ".claude/policy_finder.json").exists()
    )


def versions(instance: str) -> list:
    """An instance's versions in order: its name, then <name>-v<k> by k. A
    version counts once it has a rule or once the launcher started it (its
    instance file names it), so a version still being designed is listed."""
    names = [r.stem for r in (WT_ROOT / instance / RULE.format(name="x")).parent.glob(
        f"{instance}-v*.yml"
    )]
    try:
        config = json.loads((WT_ROOT / instance / ".claude/policy_finder.json").read_text())
        names.append(config.get("rule_name") or instance)
    except (OSError, ValueError):
        pass
    later = set()
    for name in names:
        m = LATER.match(name)
        if m and m["instance"] == instance:
            later.add((int(m["k"]), name))
    return [instance] + [v for _, v in sorted(later)]


def status(data: dict):
    todo = []
    for name in (v for i in instances() for v in versions(i)):
        sha = rule_sha(name)
        if sha is None:
            continue  # no rule (yet): nothing to summarize
        entry = data["instances"].get(name)
        if entry is None:
            todo.append((name, "new"))
        elif entry.get("rule_sha") != sha:
            todo.append((name, "rule changed"))
        elif not entry.get("policy_category"):
            todo.append((name, "no policy category yet"))
    done = len(data["instances"])
    print(f"{SUMMARY}: {done} recorded, {len(todo)} to summarize")
    print("categories:")
    for cat, spec in data["categories"].items():
        print(f"  [{spec['kind']}] {cat}: {spec['meaning']}")
    for name, why in todo:
        swept = sweep_path(name).exists()
        print(f"\n{name} ({why}{'' if swept else ', not swept yet'})")
        print(f"  rule:  {rule_path(name)}")
        if swept:
            print(f"  sweep: {sweep_path(name)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("pull")
    sub.add_parser("push")
    rec = sub.add_parser("record")
    rec.add_argument("name")
    rec.add_argument("--rule", action="append", required=True)
    rec.add_argument("--policy-category", required=True)
    rec.add_argument("--idea", required=True)
    cat = sub.add_parser("category")
    cat.add_argument("name")
    cat.add_argument("--kind", choices=KINDS, required=True)
    cat.add_argument("--meaning", required=True)
    args = ap.parse_args()

    if args.command == "status":
        pull()
        status(load())
        return
    if args.command in ("pull", "push"):
        (pull if args.command == "pull" else push)()
        return
    # instances finishing together each record: one writer at a time
    with open(SUMMARY.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write(args)


def write(args):
    """`category` or `record` on the kept copy, under the caller's lock."""
    pull()
    data = load()
    if args.command == "category":
        kind = KINDS[args.kind]
        data["categories"][args.name] = {"kind": kind, "meaning": args.meaning}
        save(data)
        print(f"category {args.name} ({kind})")
        push()
    else:
        sha = rule_sha(args.name)
        if sha is None:
            sys.exit(f"no rule for {args.name}")
        kinds = {c: s["kind"] for c, s in data["categories"].items()}
        named = [*args.rule, args.policy_category]
        unknown = [c for c in named if c not in kinds]
        if unknown:
            sys.exit(f"unknown categories {unknown}: add them with `category` first")
        if any(kinds[r] != "rule" for r in args.rule) or (
            kinds[args.policy_category] != "policy category"
        ):
            sys.exit("--rule takes a rule, --policy-category a policy category")
        data["instances"][args.name] = {
            "rule_sha": sha,
            "summarized": date.today().isoformat(),
            "rules_included": args.rule,
            "policy_category": args.policy_category,
            "idea": args.idea,
        }
        save(data)
        print(f"recorded {args.name}")
        push()


if __name__ == "__main__":
    main()
