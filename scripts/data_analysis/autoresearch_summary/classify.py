"""Guard rails around the method classification (summary doc section 4).

The classifier itself is pluggable (a model reading ONLY the blinded
input file); this script prepares that input and validates the output.

Usage:
    python .../classify.py --prepare
        experiments.json -> data/classification_input.json
        ({pr, method_text} pairs + the taxonomy; [SUCCESS]/[FAIL]
        tokens stripped so ancestor mentions carry no verdicts)

    python .../classify.py --merge <assignments.json>
        validates {pr, category, rationale} records against the
        taxonomy and coverage rules, then writes data/categories.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"

TAXONOMY = {
    "correlated-sampling": "copulas / shared latents applied at sampling time",
    "persistent-latent": "training-time latent variables (agent or group types)",
    "nonlinear-emission": "MLP / XGBoost / regression emission heads",
    "autoregressive": "observed-history conditioning",
    "structured-head": "joint or structured decision heads",
    "feature-engineering": "new input features on an unchanged model",
    "training-regime": "curriculum / sampling-schedule changes",
    "architecture": "graph / attention structure changes",
    "other": "none of the above (3+ of these force a taxonomy revision)",
}
MAX_OTHER = 2


def prepare():
    experiments = json.loads((DATA / "experiments.json").read_text())
    items = [
        {
            "pr": e["pr"],
            "method_text": re.sub(
                r"\[?\b(SUCCESS|FAIL)\b\]?", "", e["declaration_method"]
            ).strip(),
        }
        for e in experiments
    ]
    out = DATA / "classification_input.json"
    out.write_text(json.dumps({"taxonomy": TAXONOMY, "items": items}, indent=2))
    print(f"wrote {out} ({len(items)} items)")


def merge(assignments_path):
    experiments = json.loads((DATA / "experiments.json").read_text())
    expected = {e["pr"] for e in experiments}
    assignments = json.loads(Path(assignments_path).read_text())

    errors = []
    seen = set()
    for a in assignments:
        pr = a.get("pr")
        if pr in seen:
            errors.append(f"#{pr}: assigned twice")
        seen.add(pr)
        if a.get("category") not in TAXONOMY:
            errors.append(f"#{pr}: unknown category {a.get('category')!r}")
        if not a.get("rationale", "").strip():
            errors.append(f"#{pr}: empty rationale")
    missing = expected - seen
    extra = seen - expected
    if missing:
        errors.append(f"uncovered PRs: {sorted(missing)}")
    if extra:
        errors.append(f"unknown PRs: {sorted(extra)}")
    others = [a["pr"] for a in assignments if a.get("category") == "other"]
    if len(others) > MAX_OTHER:
        errors.append(
            f"{len(others)} PRs in 'other' ({others}): the taxonomy "
            "needs a new class (maintainer decision)"
        )

    if errors:
        for e in errors:
            print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)

    out = DATA / "categories.json"
    out.write_text(json.dumps(sorted(assignments, key=lambda a: a["pr"]),
                              indent=2) + "\n")
    counts = {}
    for a in assignments:
        counts[a["category"]] = counts.get(a["category"], 0) + 1
    print(f"wrote {out}")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--merge", metavar="ASSIGNMENTS_JSON")
    args = ap.parse_args()
    prepare() if args.prepare else merge(args.merge)
