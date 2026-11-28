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

An instance with harness opencode (instance_settings.py) gets its agent as
OpenCode reads it instead: `.opencode/opencode.json` (the OpenRouter model,
the key from the environment, the upstream providers) and
`.opencode/agents/policy-finder.md`, the same definition with OpenCode's
frontmatter and tool names (`opencode_agent`). OpenCode loads AGENTS.md and
CLAUDE.md into the context from the working directory up; the export holds
neither (the sparse checkout leaves CLAUDE.md out), and the build removes any
that turns up.

A single-idea instance's export also holds `test_data/`: scrambled copies of
its data files (scramble_data.py), read-only, which the agent debugs its one
analysis script on.

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
import glob
import hashlib
import json
import math
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
    "configs/simulation/manager_testing/33_MULTI_h2h",
    "configs/simulation/manager_testing/34_DIFF_h2h",
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
            "plots/simulation/29_EXP1_reference",
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
#: The agent definition of an instance whose `mode` is not free
MODE_AGENTS = {"single_idea": ".claude/agents/policy-finder-single-idea.md"}
CLAUDE_KEEP = ("agents/policy-finder.md", "agents/policy-finder-host.md", "policy_finder.json")
#: An instance's name, as new_instance.sh and the campaign make them
INSTANCE_NAME = re.compile(r"pf-\d{8}-\d+")


def sibling_pattern(name):
    """Names of instances of the same campaign as `name` (its prefix and any
    lineage number), and of the reference campaigns' (INSTANCE_NAME): a
    campaign's prefix need not look like pf-<date> (a DAIS campaign's prefix names
    its experiment and date)."""
    prefix = re.sub(r"-\d+$", "", name)
    if prefix == name:
        return INSTANCE_NAME
    return re.compile(rf"{INSTANCE_NAME.pattern}|(?<![\w-]){re.escape(prefix)}-\d+")


#: Harness opencode: the folder OpenCode loads from the working directory,
#: the agent's wording that names Claude Code's tools (OpenCode's are
#: lower-case), and the rule files OpenCode would load into the context
OPENCODE_DIR = ".opencode"
OPENCODE_WORDING = ((re.compile(r"the(\s+)Edit(\s+)tool"), r"the\1edit\2tool"),)
CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md")
#: OpenCode (1.19.6) writes this .gitignore into every config folder it loads
#: and fails when it cannot; the export's folder is read-only, so it is
#: written here with the same content, which OpenCode then leaves alone
OPENCODE_GITIGNORE = "node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore"
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


def opencode_agent(text):
    """A Claude Code agent definition as an OpenCode primary agent.

    The body stays as it is but for the tool names of OPENCODE_WORDING; the
    frontmatter keeps the description (without the relay that starts the
    Claude agent). Model and permissions are in opencode_config.
    """
    head, sep, body = text.partition("\n---\n")
    if not head.startswith("---\n") or not sep:
        raise ExportError("the agent definition has no frontmatter")
    found = re.search(r"(?m)^description: (.*)$", head)
    if not found:
        raise ExportError("the agent definition has no description")
    description = re.sub(r"[,;]? spawned by policy-finder-host", "", found.group(1))
    for pattern, replacement in OPENCODE_WORDING:
        body, n = pattern.subn(replacement, body)
        if not n:
            raise ExportError(f"the agent definition no longer says {pattern.pattern!r}")
    head = f"---\ndescription: {json.dumps(description)}\nmode: primary\n"
    return head + sep + body


#: A server of our own (base_url): the model card's sampling settings and
#: its output limit, by served name. OpenCode sends the sampling options with
#: every request. The context is the server's (PF_LLM_CONTEXT, which
#: infra/dais/campaign.slurm.sh sets from MAX_MODEL_LEN; the value here when
#: unset), so that OpenCode compacts a long session before the server refuses.
LOCAL_MODELS = {
    # huggingface.co/Qwen/Qwen3-Coder-Next, "Best Practices" (also its
    # generation_config.json at revision a7fbcb5c); 262,144 tokens natively
    "Qwen/Qwen3-Coder-Next": {
        "options": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
        "limit": {"context": 262144, "output": 16384},
    },
    # huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 (the dated card, at
    # revision 08c8657f), "Recommended sampling parameters": temperature 1.0,
    # top_p 1.0, max_tokens >= 256K. The maximum reasoning effort is the
    # server's default (infra/dais/model_profiles/deepseek-v4.1-flash-1017.sh),
    # as is the context (1M tokens).
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": {
        "options": {"temperature": 1.0, "top_p": 1.0},
        "limit": {"context": 1048576, "output": 262144},
    },
    # huggingface.co/moonshotai/Kimi-K3.1-Pro (revision 80aff651), "Evaluation
    # Results": temperature 1.0, top_p 0.95, reasoning effort max; its
    # generation_config.json sets no sampling. The effort is the server's
    # default (infra/dais/model_profiles/kimi-k3.1-pro.sh), as is the context
    # (1M tokens). The card states no output limit: 131072 here.
    "moonshotai/Kimi-K3.1-Pro": {
        "options": {"temperature": 1.0, "top_p": 0.95},
        "limit": {"context": 1048576, "output": 131072},
    },
}
#: Tokens OpenCode keeps free before it compacts (its `compaction.reserved`),
#: beyond the output limit: room for the tool results one step adds (one
#: tool output is capped at 50 KB or 2000 lines, 20-25k tokens of numeric
#: text; a step may make two) and for the compaction request itself. Without
#: it OpenCode sent a request past the server's context, the server refused
#: it, and the compaction that followed failed the same way (9 runs of the
#: first Qwen campaigns ended so).
COMPACTION_MARGIN = 65536


def local_model(model):
    """A local model's OpenCode entry and the compaction setting for it."""
    entry = {"name": model, **LOCAL_MODELS.get(model, {})}
    limit = dict(entry.get("limit") or {"context": 131072, "output": 16384})
    context = os.environ.get("PF_LLM_CONTEXT")
    if context:
        try:
            limit["context"] = int(context)
        except ValueError:
            raise ExportError(f"PF_LLM_CONTEXT is not a number of tokens: {context!r}") from None
    # OpenCode 1.19.6 (session/overflow.ts) applies `compaction.reserved` only
    # to a model with `limit.input`; without one it compacts at context minus
    # the output limit, with no margin at all
    limit["input"] = limit["context"]
    entry["limit"] = limit
    reserved = limit["output"] + COMPACTION_MARGIN
    if limit["context"] - reserved < 32768:  # too little left to work in
        raise ExportError(f"context {limit['context']} is too small for output "
                          f"{limit['output']} and a compaction margin of {COMPACTION_MARGIN}")
    return entry, {"auto": True, "reserved": reserved}


def opencode_config(conf):
    """The instance's OpenCode config: its model, nothing else.

    On OpenRouter by default; with `base_url` in the instance settings, on
    that OpenAI-compatible server instead (a vLLM endpoint, provider id
    `local`, its key in PF_LLM_API_KEY, `model` the served name). The
    image's global config (infra/claude_proxy/container/CC_env/opencode/)
    holds the permissions and turns off what would call out. The key comes
    from the agent's environment and is never written here.
    """
    model = conf["model"]
    if conf.get("base_url"):
        entry, compaction = local_model(model)
        return {
            "$schema": "https://opencode.ai/config.json",
            "enabled_providers": ["local"],
            "model": f"local/{model}",
            "small_model": f"local/{model}",
            "compaction": compaction,
            "provider": {"local": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": conf["base_url"],
                            "apiKey": "{env:PF_LLM_API_KEY}"},
                "models": {model: entry},
            }},
        }
    options = {}
    if conf.get("providers"):
        options["provider"] = {"order": list(conf["providers"]), "allow_fallbacks": False}
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"openrouter/{model}",
        "small_model": f"openrouter/{model}",
        "provider": {"openrouter": {
            "options": {"apiKey": "{env:OPENROUTER_API_KEY}"},
            "models": {model: {"options": options}},
        }},
    }


#: The sweep results an agent reads, under each earlier version's sweep
#: folder: written for the agent in a compact form (slim_sweep_files)
SWEEP_FILES = ("sweep.json", "landscapes.json", "final/sweep.json")


def _round(x, digits=4):
    """`x` with every float rounded to `digits` significant digits."""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}") if math.isfinite(x) else x
    if isinstance(x, dict):
        return {k: _round(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, digits) for v in x]
    return x


#: OpenCode's read tool returns at most 2000 lines or 50 KB of a file
READ_LIMIT_BYTES = 50 * 1000
READ_LIMIT_LINES = 2000


#: OpenCode's read tool also cuts every line at 2000 characters
READ_LIMIT_LINE_CHARS = 2000
#: Lines this file writes stay under it, with room to spare
LINE_CHARS = 1500


def _dump(x):
    """Compact JSON of `x` in lines of at most LINE_CHARS where it can: a
    dict or list too long for one line puts its items on lines of their
    own, short items grouped (only an atom longer than that, a long
    string, keeps a long line)."""
    one = json.dumps(x, separators=(",", ":"))
    if len(one) <= LINE_CHARS or not isinstance(x, (dict, list)) or not x:
        return one
    if isinstance(x, dict):
        items = [json.dumps(k) + ":" + _dump(v) for k, v in x.items()]
        opening, closing = "{", "}"
    else:
        items = [_dump(v) for v in x]
        opening, closing = "[", "]"
    lines, line = [], ""
    for item in items:
        if line and ("\n" in item or len(line) + 1 + len(item) > LINE_CHARS):
            lines.append(line)
            line = item
        else:
            line = f"{line},{item}" if line else item
    lines.append(line)
    return opening + ",\n".join(lines) + closing


def slim_json(data, rows="points", sort_by=None):
    """A sweep file as the agent gets it: the same keys and entries, floats to
    4 significant digits (fewer would merge the points' scores: exp1's pools
    of 630 to 660 at 2 digits take 4 distinct values), compact, one entry of
    `rows` per line (sorted by `sort_by`, highest first, when given), and no
    line over LINE_CHARS where it can be broken (the read tool cuts lines at
    2000 characters)."""
    head = {k: v for k, v in data.items() if k != rows}
    entries = list(data.get(rows) or [])
    if sort_by and all(isinstance(e.get(sort_by), (int, float)) for e in entries):
        entries.sort(key=lambda e: -e[sort_by])
    compact = _dump(_round(head))
    if rows not in data:
        return compact + "\n"
    lines = ",\n".join(_dump(_round(e)) for e in entries)
    sep = ",\n" if head else ""
    return f'{compact[:-1]}{sep}"{rows}":[\n{lines}\n]}}\n'


def slim_sweep_files(export):
    """Rewrite the earlier versions' sweep results in the export compactly:
    the agent reads them whole (a pretty-printed sweep.json of 256 points was
    over 3,000 lines and 90k characters, cut by the read tool at 50 KB). The
    worktree keeps the full files for the pipeline (fit_surrogate.py,
    generate_sim_config.py --final, merge_sweeps.py, reevaluate.py), and
    collect() never copies these back: they are outside the write folders.
    Returns {path: (characters before, after)}."""
    sizes = {}
    root = os.path.join(export, "plots/simulation/policy_finder")
    for sweep in sorted(glob.glob(os.path.join(root, "*_sweep"))):
        for rel in SWEEP_FILES:
            path = os.path.join(sweep, rel)
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as f:
                    text = f.read()
                data = json.loads(text)
            except (OSError, ValueError):
                continue  # not ours to judge: left as it is
            sort_by = "pool" if rel.endswith("sweep.json") else "mean"
            slim = slim_json(data, "points", sort_by)
            if (len(slim.encode()) > READ_LIMIT_BYTES
                    or slim.count("\n") >= READ_LIMIT_LINES):
                # the read tool cuts the tail, the lowest-scoring points
                # (best first); fewer digits would merge the scores instead
                print(f"WARNING: {os.path.relpath(path, export)} is {len(slim.encode())} "
                      f"bytes, over the agent's read limit of {READ_LIMIT_BYTES}: its "
                      f"lowest points are cut when read", file=sys.stderr, flush=True)
            with open(path, "w") as f:
                f.write(slim)
            sizes[os.path.relpath(path, export)] = (len(text), len(slim))
    return sizes


def write_opencode(export, conf, agent_text):
    folder = os.path.join(export, OPENCODE_DIR)
    os.makedirs(os.path.join(folder, "agents"), exist_ok=True)
    with open(os.path.join(folder, "opencode.json"), "w") as f:
        json.dump(opencode_config(conf), f, indent=2)
        f.write("\n")
    Path(folder, "agents", "policy-finder.md").write_text(opencode_agent(agent_text))
    Path(folder, ".gitignore").write_text(OPENCODE_GITIGNORE)
    for root, _, files in os.walk(export):
        for f in files:
            if f in CONTEXT_FILES:
                os.remove(os.path.join(root, f))


def leaks(export, name, terms=()):
    """Text files that name another instance, or match one of `terms`
    (regular expressions): [(path, match)]."""
    patterns = [re.compile(term) for term in terms]
    siblings = sibling_pattern(name)
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
            for match in set(m.group(0) for m in siblings.finditer(text)):
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
    # a single-idea instance has its own agent (instance_settings.py)
    agent_path = MODE_AGENTS.get(conf.get("mode", "free"), setting["agent"])
    agent = os.path.join(export, agent_path)
    target = os.path.join(export, ".claude/agents/policy-finder.md")
    if not os.path.isfile(agent):
        raise ExportError(f"no agent definition {agent_path} in {wt}")
    if agent != target:
        # policy-finder-host starts the subagent by its frontmatter name
        text = Path(agent).read_text()
        head, sep, body = text.partition("\n---\n")
        head = re.sub(r"(?m)^name: .*$", "name: policy-finder", head, count=1)
        Path(target).write_text(head + sep + body)
    if conf.get("harness") == "opencode":
        write_opencode(export, conf, Path(agent).read_text())
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
    slim_sweep_files(export)
    if conf.get("mode") == "single_idea":
        # scrambled copies the agent debugs its one analysis script on
        import scramble_data

        scramble_data.build(export)
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
