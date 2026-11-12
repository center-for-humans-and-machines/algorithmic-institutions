"""Run directory, file helpers, config, quote check. Standard library + yaml."""

import copy
import json
import os
import re
import sys
import uuid
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/annotation/default.yml"


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


def load_config(path=None):
    cfg = read_yaml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    if path:
        cfg = deep_merge(cfg, read_yaml(path))
    return cfg


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


def quote_in_source(quote, source):
    """Whether a quote occurs in its source, whitespace-normalised.

    Quotes elided with "..." pass if every part occurs. Empty quote -> None.
    """
    q = _norm(quote).strip("\"'`")
    if not q:
        return None
    src = _norm(source)
    parts = [p.strip() for p in re.split(r"\s*(?:\.\.\.|…)\s*", q) if p.strip()]
    return all(p in src for p in parts) if parts else None


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "c_" + s
    return s[:60]


_REV = re.compile(r"^(what|why)\.v(\d+)\.yml$")


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


def latest_codebook(run, kind):
    """(revision, path, codes) of the highest codebook revision."""
    revs = codebook_revisions(run, kind)
    if not revs:
        raise SystemExit(f"no codebook/{kind}.v*.yml in {run.dir}; run stage A")
    rev, path = revs[-1]
    return rev, path, read_yaml(path)["codes"]
