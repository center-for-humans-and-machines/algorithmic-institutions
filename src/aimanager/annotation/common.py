"""Run directory, file helpers, config, quote check. Standard library + yaml."""

import copy
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path

import yaml

# The package runs from a plain source tree (PYTHONPATH=src, no install), so the
# default config is found relative to this file, or named by ANNOT_CONFIG.
REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs/annotation/default.yml"


class StepFailed(SystemExit):
    """A step could not produce valid outputs; nothing was written. Exit code 1."""

    def __init__(self, msg):
        super().__init__(msg)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for key, val in (extra or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def default_config_path():
    env = os.environ.get("ANNOT_CONFIG")
    path = Path(env) if env else REPO_CONFIG
    if not path.exists():
        raise SystemExit(
            f"default config not found at {path}; run from a source tree that "
            "has configs/annotation/default.yml, or set ANNOT_CONFIG"
        )
    return path


def load_config(path=None):
    cfg = read_yaml(default_config_path()) or {}
    if path:
        cfg = deep_merge(cfg, read_yaml(path))
    return cfg


def check_failures(cfg, step, kind, failed, total):
    """Raise StepFailed if more calls of a pass kind failed than the budget allows.

    `failed` is a list of (item_id, error). Called before a step writes anything.
    """
    if not failed:
        return
    share = (cfg.get("failure_budget") or {}).get(kind, 0.0)
    allowed = math.floor(share * total + 1e-9)
    if len(failed) <= allowed:
        log(f"  {step}: {len(failed)}/{total} {kind} calls failed, within budget")
        return
    first = "; ".join(f"{i}: {e[:160]}" for i, e in failed[:3])
    raise StepFailed(
        f"{step}: {len(failed)} of {total} {kind} calls failed, budget "
        f"failure_budget.{kind} = {share} allows {allowed}. Nothing was written. "
        f"First errors: {first}"
    )


class Run:
    """Paths of one run directory: <runs_root>/<run_id>/."""

    def __init__(self, runs_root, run_id):
        self.dir = Path(runs_root) / run_id
        self.run_id = run_id

    def path(self, *parts):
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(path, text)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _Dumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_presenter)


def dump_yaml(obj):
    return yaml.dump(obj, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=88)


def write_yaml(path, obj):
    atomic_write_text(path, dump_yaml(obj))


def read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _norm_quote(text):
    """Whitespace, case and markdown emphasis (`, *) do not count as changes."""
    return _norm(re.sub(r"[`*]", "", text or "")).casefold()


def quote_in_source(quote, source):
    """Whether a quote occurs in its source.

    Compared after normalising whitespace, case and markdown emphasis marks (a
    quote that only capitalises its first word or drops backticks is verbatim).
    Quotes elided with "..." pass if every part occurs. Empty quote -> None.
    """
    q = _norm_quote(quote).strip("\"'")
    if not q:
        return None
    src = _norm_quote(source)
    parts = [p.strip() for p in re.split(r"\s*(?:\.\.\.|…)\s*", q) if p.strip()]
    return all(p in src for p in parts) if parts else None


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- manifests ----------------------------------------------------------------


def _rel(run, path):
    return str(Path(path).resolve().relative_to(run.dir.resolve()))


def write_manifest(run, step, inputs, outputs, **info):
    """manifests/<step>.json, written last: inputs and outputs with hashes.

    `info` holds expected and written counts, failed items, codebooks, model.
    """
    doc = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {_rel(run, p): file_hash(p) for p in sorted(set(map(Path, inputs)))},
        "outputs": {
            _rel(run, p): file_hash(p) for p in sorted(set(map(Path, outputs)))
        },
        **info,
    }
    write_json(run.path("manifests", f"{step}.json"), doc)
    return doc


def require(run, step):
    """The manifest of a finished step, after checking it is still current.

    Fails if the step never finished, if one of its outputs changed or vanished
    since (another run of something overwrote it), or if one of its inputs
    changed since (an upstream step was rerun, so this step is stale).
    """
    path = run.dir / "manifests" / f"{step}.json"
    if not path.exists():
        raise SystemExit(f"step {step} has not finished in {run.dir}; run it first")
    doc = read_json(path)
    for kind, files in (("output", doc["outputs"]), ("input", doc["inputs"])):
        for rel, digest in files.items():
            p = run.dir / rel
            if not p.exists() or file_hash(p) != digest:
                what = "changed or vanished" if kind == "output" else "changed"
                raise SystemExit(
                    f"step {step} is stale: its {kind} {rel} {what} after it ran; "
                    f"rerun {step} (and the steps after it)"
                )
    return doc


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "c_" + s
    return s[:60]


_REV = re.compile(r"^(what|why|basis)\.v(\d+)\.yml$")


def codebook_revisions(run, kind):
    """Sorted [(revision, path)] of codebook/<kind>.v<k>.yml in a run."""
    d = run.dir / "codebook"
    out = []
    if d.exists():
        for p in d.iterdir():
            m = _REV.match(p.name)
            if m and m.group(1) == kind:
                out.append((int(m.group(2)), p))
    return sorted(out)


CODEBOOK_STEP = {
    "what": "consolidate-what",
    "why": "consolidate-why",
    "basis": "basis-codebook",
}


def check_revision_chain(run, kind):
    """[(revision, path)] after checking every revision against its base.

    v1 must be the file its producing step wrote. Every later revision must
    carry `based_on: {revision: k-1, sha256: <hash of v(k-1)>}`, so a review
    written against an older v1 is rejected once v1 is regenerated.
    """
    revs = codebook_revisions(run, kind)
    if not revs:
        raise SystemExit(f"no codebook/{kind}.v*.yml in {run.dir}; run stage A")
    if revs[0][0] != 1 or [r for r, _ in revs] != list(range(1, len(revs) + 1)):
        raise SystemExit(f"codebook/{kind}: revisions must be v1, v2, ... without gaps")
    step = CODEBOOK_STEP[kind]
    rel = f"codebook/{kind}.v1.yml"
    made = require(run, step)["outputs"].get(rel)
    if made != file_hash(revs[0][1]):
        raise SystemExit(f"{rel} is not the file {step} wrote; rerun {step}")
    for (prev, ppath), (rev, path) in zip(revs, revs[1:]):
        base = (read_yaml(path) or {}).get("based_on") or {}
        if base.get("revision") != prev or base.get("sha256") != file_hash(ppath):
            raise SystemExit(
                f"codebook/{path.name} was reviewed against a different "
                f"{ppath.name} (based_on {base or 'missing'}, current sha256 "
                f"{file_hash(ppath)}). Redo the review from the current "
                f"{ppath.name} (`review begin`), or delete {path.name}"
            )
    return revs


def latest_codebook(run, kind):
    """(revision, path, codes) of the highest checked codebook revision."""
    rev, path = check_revision_chain(run, kind)[-1]
    return rev, path, read_yaml(path)["codes"]


def codebook_families(path):
    """{family: definition} of a codebook file (empty if it has no families)."""
    doc = read_yaml(path) or {}
    return {f["name"]: f.get("definition", "") for f in doc.get("families") or []}
