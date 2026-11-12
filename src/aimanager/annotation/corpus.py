"""Step 0: read the policy-finder lineages from git refs, write the corpus.

Reads with git ls-tree / git show only, no checkout. Writes
corpus/versions.jsonl and corpus/split.json.
"""

import json
import os
import random
import re
import subprocess
from collections import defaultdict

from .common import log, read_jsonl, require, write_json, write_jsonl, write_manifest


def _git(repo, *args):
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    return out.stdout


def list_refs(repo, patterns):
    refs = set()
    for pat in patterns:
        out = _git(repo, "for-each-ref", "--format=%(refname:short)", _full(pat))
        refs.update(line.strip() for line in out.splitlines() if line.strip())
    return sorted(refs, key=_lineage_key)


def _full(pat):
    if pat.startswith("refs/"):
        return pat
    return "refs/remotes/" + pat if pat.startswith("origin/") else "refs/heads/" + pat


def _lineage_key(ref):
    name = ref.rsplit("/", 1)[-1]
    m = re.match(r"^(.*)-(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def _version_of(stem, name):
    if stem == name:
        return 1
    m = re.match(rf"^{re.escape(name)}-v(\d+)$", stem)
    return int(m.group(1)) if m else None


def read_score(repo, ref, path):
    """(score, score_se, status) from a final sweep.json, or (None, None, None)."""
    try:
        text = _git(repo, "show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None, None, None
    try:
        sweep = json.loads(text)
        best = next(p for p in sweep["points"] if p["name"] == sweep["best_name"])
        pool = best["pool"]
        if pool is None or pool != pool:
            return None, None, "failed"
        return float(pool), best.get("pool_se"), "ok"
    except (ValueError, KeyError, StopIteration, TypeError):
        return None, None, "failed"


def read_lineage(repo, ref, cfg):
    name = ref.rsplit("/", 1)[-1]
    campaign = name.rsplit("-", 1)[0]
    rule_dir, notes_dir = cfg["rule_dir"], cfg["notes_dir"]
    files = _git(repo, "ls-tree", "-r", "--name-only", ref, "--", rule_dir, notes_dir)
    own = re.compile(rf"^{re.escape(name)}(-v\d+)?\.(yml|md)$")
    stems = defaultdict(dict)
    for f in files.splitlines():
        base = f.rsplit("/", 1)[-1]
        if not own.match(base):
            continue
        stem, ext = base.rsplit(".", 1)
        if f == f"{rule_dir}/{base}" and ext == "yml":
            stems[stem]["rule"] = f
        elif f == f"{notes_dir}/{base}" and ext == "md":
            stems[stem]["notes"] = f
    rows = []
    for stem, paths in stems.items():
        rows.append(
            {
                "stem": stem,
                "campaign": campaign,
                "lineage": name,
                "version": _version_of(stem, name),
                "ref": ref,
                "rule": (
                    _git(repo, "show", f"{ref}:{paths['rule']}")
                    if "rule" in paths
                    else None
                ),
                "notes": (
                    _git(repo, "show", f"{ref}:{paths['notes']}")
                    if "notes" in paths
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: r["version"])
    for i, row in enumerate(rows):
        row["parent"] = rows[i - 1]["stem"] if i else None
        score, se, status = read_score(
            repo, ref, cfg["sweep_path"].format(stem=row["stem"])
        )
        override = cfg.get("score_status_overrides", {}).get(row["stem"])
        row["score"], row["score_se"] = score, se
        row["score_status"] = status or override or "missing"
    return rows


def make_split(versions, cfg):
    """Free-coding sample (per lineage, alternating early/late) and expert check."""
    rng = random.Random(cfg["seed"])
    by_lineage = defaultdict(list)
    for v in versions:
        by_lineage[v["lineage"]].append(v)
    lineages = sorted(by_lineage)
    rng.shuffle(lineages)
    k = cfg.get("free_coding_per_lineage", 1)
    free = []
    for i, lin in enumerate(lineages):
        vs = sorted(by_lineage[lin], key=lambda v: v["version"])
        order = vs if i % 2 == 0 else vs[::-1]
        free.extend(v["stem"] for v in order[:k])
    rest = defaultdict(list)
    for v in versions:
        if v["stem"] not in free:
            rest[v["campaign"]].append(v["stem"])
    n_rest = sum(len(s) for s in rest.values())
    n_target = min(cfg.get("expert_check_n", 30), n_rest)
    expert = []
    campaigns = sorted(rest)
    # proportional allocation, largest remainders
    quotas = {c: n_target * len(rest[c]) / max(n_rest, 1) for c in campaigns}
    alloc = {c: int(q) for c, q in quotas.items()}
    for c in sorted(campaigns, key=lambda c: alloc[c] - quotas[c]):
        if sum(alloc.values()) >= n_target:
            break
        alloc[c] += 1
    for c in campaigns:
        expert.extend(sorted(rng.sample(sorted(rest[c]), alloc[c])))
    return {
        "seed": cfg["seed"],
        "free_coding": free,
        "expert_check": expert,
    }


def run_corpus(run, cfg, limit=None, repo=None):
    ccfg = cfg["corpus"]
    repo = repo or ccfg["repo"]
    refs = list_refs(repo, ccfg["refs"])
    if limit:
        refs = refs[:limit]
    versions = []
    for ref in refs:
        versions.extend(read_lineage(repo, ref, ccfg))
    settings = cfg.get("settings") or {}
    missing = sorted({v["campaign"] for v in versions} - set(settings))
    if missing:
        raise SystemExit(f"no game setting in the config for campaigns {missing}")
    for v in versions:
        v["setting"] = settings[v["campaign"]].strip()
        if v["rule"] is None or v["notes"] is None:
            log(f"warning: {v['stem']} lacks a rule or notes file")
    split = make_split(versions, cfg["sampling"])
    out = [run.path("corpus", "versions.jsonl"), run.path("corpus", "split.json")]
    write_jsonl(out[0], versions)
    write_json(out[1], split)
    commits = {ref: _git(repo, "rev-parse", ref).strip() for ref in refs}
    write_manifest(
        run,
        "corpus",
        [],
        out,
        expected=len(versions),
        written=len(versions),
        refs=commits,
    )
    n_ok = sum(v["score_status"] == "ok" for v in versions)
    log(
        f"corpus: {len(refs)} refs, {len(versions)} versions, {n_ok} scored; "
        f"free coding {len(split['free_coding'])}, "
        f"expert check {len(split['expert_check'])}"
    )
    return versions


def load_versions(run):
    require(run, "corpus")
    return read_jsonl(run.dir / "corpus" / "versions.jsonl")
