"""Build what a policy-finder agent sees, and take its work back (#236, #244).

An agent runs in its own container (agent_runner.py), which mounts one export
of its instance: a plain copy of the instance worktree's current state with
no git, read-only except the version's write paths. The container is the only
sandbox, so the export decides what the agent can see.

`build` copies into `<exports>/<stem>/`:
  - the worktree (its sparse checkout, scripts/policy_finder/instance_paths:
    code, configs, data, the game rules and the reference sims, and from
    version 2 on the lineage's earlier versions with their sweeps and final
    runs), without `.git`;
  - AGENT_DATA from the main checkout: the data files the agent reads, which
    are not in git (docs/large_files.md);
then drops what tells of other instances or of earlier results (DENY and
every rule, notes, sim config or scripts folder of another lineage), and keeps
from `.claude/` only the agent definitions and the instance file. A leak check
then fails the build if any other instance's name (`pf-<date>-<n>`) is left in
a text file. Last, the version's write folders are made (they are mounted
read-write) and every file in them is hashed into `.export_manifest.json`,
outside the export.

`collect` copies the version's rule and notes and the instance's scripts
folder back into the worktree, and returns the files in the write folders
that the version may not touch but did: a new file other than its rule, notes
or scripts, or a changed or deleted file that was there (an earlier
version's rule, say). The pipeline fails the version on any.

Usage:
    python scripts/policy_finder/export_instance.py build <worktree> <main> <exports>
    python scripts/policy_finder/export_instance.py collect <worktree> <exports>
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

#: Not in git (docs/large_files.md): the human data and the reference sims the
#: agent reads, copied from the main checkout. The same files the old launcher
#: pulled from LFS; the AH models and other large files stay out, as then.
AGENT_DATA = (
    "experiments/2group_8agent_50ep.csv",
    "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/*",
)
#: Never in an export: earlier results and the tools that list instances.
DENY = (
    "configs/simulation/manager_testing/26_LEVIN_ah_vs_pf_top_batched.yml",
    "configs/simulation/manager_testing/27_LEVIN_pf_proxy_vs_cem_batched.yml",
    "configs/simulation/manager_testing/28_EXP1_baselines.yml",
    "configs/simulation/manager_testing/30_EXP1_pf_vs_rl.yml",
    "configs/simulation/manager_testing/31_EXP1_confirm",
    "configs/simulation/opponents",
    "configs/simulation/manager_testing/32_LEVIN_multi_reference.yml",
    "scripts/policy_finder/observation_app",
    "scripts/policy_finder/campaign.py",
    "scripts/policy_finder/campaigns",
    "scripts/tests",
)
#: Per game (the instance file's `setting`, manager/rule.py SETTINGS): the
#: data the agent reads (AGENT_DATA's role), what else stays out (the other
#: game's rules and data, and for experiment 1 the RL manager it is compared
#: with), and the agent definition the export holds as
#: `.claude/agents/policy-finder.md`, the one policy-finder-host starts.
SETTINGS = {
    "exp2": {
        "agent_data": AGENT_DATA,
        "deny": (
            "docs/reports/basics_exp1.md",
            "docs/reports/basics_exp2_multi.md",
            "docs/reports/basics_exp2_diff.md",
        ),
        "agent": ".claude/agents/policy-finder.md",
    },
    "exp1": {
        "agent_data": (
            "experiments/pilot_random1_player_round_slim.csv",
            "plots/simulation/29_EXP1_reference/*",
        ),
        "deny": (
            "docs/reports/basics.md",
            "docs/reports/basics_exp2_multi.md",
            "docs/reports/basics_exp2_diff.md",
            "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched",
            "artifacts/manager",
        ),
        "agent": ".claude/agents/policy-finder-exp1.md",
    },
    # experiment 2 against several governors: the agent learns nothing of them
    # up front (no game doc naming one, no reference sim of any governor, no
    # sim config, no opponents file) and sees its own results under opaque keys
    "exp2_multi": {
        "agent_data": ("experiments/2group_8agent_50ep.csv",),
        "deny": (
            "docs/reports/basics.md",
            "docs/reports/basics_exp1.md",
            "docs/reports/basics_exp2_diff.md",
            "plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched",
            "plots/simulation/29_EXP1_reference",
            "configs/simulation/manager_testing",
            "configs/simulation/opponents",
            "configs/simulation/policy_finder",
            "configs/training/rl_manager",
            "configs/training/baselines",
            # the human-manager clone, one of the governors: its tests and training
            "configs/simulation/ah_testing",
            "configs/training/artificial_humans/punishment",
            "artifacts/manager",
            "artifacts/baselines",
            "src/aimanager/tests",
            # analyses of earlier sims and of the clones' training, naming the
            # governors they played
            "scripts/plotting",
            "scripts/baselines",
            "scripts/kinit_mpcdf.sh",
        ),
        "agent": ".claude/agents/policy-finder-exp2-multi.md",
        # text that would tell an agent who its opponents are
        "leak_terms": (
            r"(?i)levin",
            r"_vs_ah\b",
            r"punishment_multinomial",
            r"sigmoid_opt",
            r"opponents/exp2_multi",
        ),
    },
    # experiment 2 scored by the lead over `ah`'s group: exp2's view (the
    # reference sims against `ah`), with its own game doc and agent
    "exp2_diff": {
        "agent_data": AGENT_DATA,
        "deny": (
            "docs/reports/basics.md",
            "docs/reports/basics_exp1.md",
            "docs/reports/basics_exp2_multi.md",
        ),
        "agent": ".claude/agents/policy-finder-exp2-diff.md",
    },
}
#: Folders that hold one entry per instance or version; only the lineage's own
#: entries stay (and .gitkeep)
PER_INSTANCE = (
    "configs/managers/rule_based",
    "configs/managers/rule_based/params",
    "configs/simulation/policy_finder",
    "notes/policy_finder",
    "plots/simulation/policy_finder",
    "scripts/policy_finder",
)
#: What stays of .claude/
CLAUDE_KEEP = ("agents/policy-finder.md", "agents/policy-finder-host.md", "policy_finder.json")
#: An instance's name, as new_instance.sh and the campaign make them
INSTANCE_NAME = re.compile(r"pf-\d{8}-\d+")
#: Text files above this size are not scanned (data files)
SCAN_LIMIT = 20 * 2**20
MANIFEST = ".export_manifest.json"


class ExportError(Exception):
    pass


def instance(wt):
    with open(os.path.join(wt, ".claude", "policy_finder.json")) as f:
        return json.load(f)


def write_dirs(name, stem):
    """The folders mounted read-write: the rule's, the notes' and the scripts'."""
    return (
        "configs/managers/rule_based",
        "notes/policy_finder",
        f"scripts/policy_finder/{name}",
    )


def write_paths(name, stem):
    """The paths the version writes (instance_settings.write_paths)."""
    return (
        f"configs/managers/rule_based/{stem}.yml",
        f"notes/policy_finder/{stem}.md",
        f"scripts/policy_finder/{name}",
    )


def _lineage(entry, name):
    """True for a per-instance entry of the lineage `name`, any version:
    `<name>`, `<name>-v<k>`, then optionally `.`/`_` and more
    (`<name>-v2.yml`, `<name>_sweep_p1of2.yml`); `<name>0...` is another
    instance."""
    return re.match(rf"^{re.escape(name)}(-v\d+)?([._].*)?$", entry) is not None


def _rm(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def _prune(export, name, deny=()):
    for rel in (*DENY, *deny):
        _rm(os.path.join(export, rel))
    # parents before children: a child folder in PER_INSTANCE is kept by its
    # parent and pruned in its own turn
    for folder in PER_INSTANCE:
        directory = os.path.join(export, folder)
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            keep = (
                entry == ".gitkeep"
                or _lineage(entry, name)
                or f"{folder}/{entry}" in PER_INSTANCE
            )
            if not keep:
                _rm(os.path.join(directory, entry))
    claude = os.path.join(export, ".claude")
    if os.path.isdir(claude):
        for root, dirs, files in os.walk(claude, topdown=False):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), claude)
                if rel not in CLAUDE_KEEP:
                    os.remove(os.path.join(root, f))
            for d in dirs:
                p = os.path.join(root, d)
                if not os.listdir(p):
                    os.rmdir(p)


def leaks(export, name, terms=()):
    """Text files that name another instance, or match one of `terms`
    (regular expressions): [(path, match)]."""
    patterns = [re.compile(term) for term in terms]
    found = []
    for root, _, files in os.walk(export):
        for f in files:
            path = os.path.join(root, f)
            try:
                if os.path.getsize(path) > SCAN_LIMIT:
                    continue
                with open(path, "rb") as handle:
                    data = handle.read()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            text = data.decode("utf-8", "replace")
            rel = os.path.relpath(path, export)
            for match in set(INSTANCE_NAME.findall(text)):
                if match != name:
                    found.append((rel, match))
            for pattern in patterns:
                hit = pattern.search(text)
                if hit:
                    found.append((rel, hit.group(0)))
    return sorted(found)


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot(export, dirs):
    files = {}
    for d in dirs:
        for root, _, names in os.walk(os.path.join(export, d)):
            for f in names:
                path = os.path.join(root, f)
                files[os.path.relpath(path, export)] = _hash(path)
    return files


def manifest_path(exports, stem):
    return os.path.join(exports, f"{stem}{MANIFEST}")


def build(wt, main, exports):
    """Build `<exports>/<stem>/` from the worktree; return its path."""
    wt, main = os.path.realpath(wt), os.path.realpath(main)
    conf = instance(wt)
    name, stem = conf["name"], conf["rule_name"]
    setting = SETTINGS[conf.get("setting", "exp2")]
    export = os.path.join(exports, stem)
    _rm(export)
    shutil.copytree(
        wt, export, symlinks=False, ignore=shutil.ignore_patterns(".git")
    )
    agent = os.path.join(export, setting["agent"])
    target = os.path.join(export, ".claude/agents/policy-finder.md")
    if not os.path.isfile(agent):
        raise ExportError(f"no agent definition {setting['agent']} in {wt}")
    if agent != target:
        # policy-finder-host starts the subagent by its frontmatter name
        text = Path(agent).read_text()
        head, sep, body = text.partition("\n---\n")
        head = re.sub(r"(?m)^name: .*$", "name: policy-finder", head, count=1)
        Path(target).write_text(head + sep + body)
    for pattern in setting["agent_data"]:
        base = os.path.dirname(pattern)
        source_dir = os.path.join(main, base)
        if not os.path.isdir(source_dir):
            raise ExportError(f"no {source_dir}: fetch it (docs/large_files.md)")
        matched = [
            f for f in sorted(os.listdir(source_dir))
            if fnmatch.fnmatch(os.path.join(base, f), pattern)
            and os.path.isfile(os.path.join(source_dir, f))
        ]
        if not matched:
            raise ExportError(f"no file in the main checkout matches {pattern}")
        os.makedirs(os.path.join(export, base), exist_ok=True)
        for f in matched:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(export, base, f))
    _prune(export, name, setting["deny"])
    found = leaks(export, name, setting.get("leak_terms", ()))
    if found:
        listing = ", ".join(f"{p} ({n})" for p, n in found[:20])
        raise ExportError(
            f"the export of {stem} names other instances or its opponents: {listing}"
        )
    dirs = write_dirs(name, stem)
    for d in dirs:
        os.makedirs(os.path.join(export, d), exist_ok=True)
    with open(manifest_path(exports, stem), "w") as f:
        json.dump({"name": name, "stem": stem, "files": _snapshot(export, dirs)}, f,
                  indent=1)
    return export


def collect(wt, exports):
    """Copy the version's write paths back into the worktree; return strays.

    A stray is a file in the write folders, other than the version's rule,
    notes and scripts folder, that is new, changed or gone since the build.
    """
    wt = os.path.realpath(wt)
    conf = instance(wt)
    name, stem = conf["name"], conf["rule_name"]
    export = os.path.join(exports, stem)
    with open(manifest_path(exports, stem)) as f:
        before = json.load(f)["files"]
    after = _snapshot(export, write_dirs(name, stem))
    rule, notes, scripts = write_paths(name, stem)

    def own(rel):
        return rel in (rule, notes) or rel.startswith(scripts + "/")

    strays = sorted(
        rel for rel in set(before) | set(after)
        if before.get(rel) != after.get(rel) and not own(rel)
    )
    for rel in (rule, notes):
        source = os.path.join(export, rel)
        if os.path.isfile(source):
            os.makedirs(os.path.dirname(os.path.join(wt, rel)), exist_ok=True)
            shutil.copy2(source, os.path.join(wt, rel))
    source = os.path.join(export, scripts)
    if os.path.isdir(source):
        target = os.path.join(wt, scripts)
        _rm(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    return strays


def remove(exports, stem):
    _rm(os.path.join(exports, stem))
    _rm(manifest_path(exports, stem))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("worktree")
    b.add_argument("main")
    b.add_argument("exports")
    c = sub.add_parser("collect")
    c.add_argument("worktree")
    c.add_argument("exports")
    args = ap.parse_args()
    try:
        if args.cmd == "build":
            print(build(args.worktree, args.main, args.exports))
        else:
            strays = collect(args.worktree, args.exports)
            for s in strays:
                print(f"outside the write paths: {s}", file=sys.stderr)
            sys.exit(1 if strays else 0)
    except ExportError as e:
        sys.exit(f"FAIL: {e}")


if __name__ == "__main__":
    main()
