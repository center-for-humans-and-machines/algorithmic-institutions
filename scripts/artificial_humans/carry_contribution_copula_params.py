"""Carry a contribution copula's calibrated rho and phi onto a retrained
marginal, WITHOUT recalibrating.

The copula's parameters are frozen per model family: when only the marginal
is retrained, the within-round latent correlation is carried over unchanged
rather than refit (the punisher side does the same through
`punishment_copula_rho.py --stamp-rho`). `make_contribution_copula_artifact.py`
refuses a params JSON whose `source_model` / `source_model_sha256` do not
match the base it is stamping, which is the right check -- so this script
writes the carried params as their own JSON, pointing at the new base and
recording, in the file, that the numbers were carried and from where.

Nothing is estimated here: rho, phi, phi_final and copula_switch_every are
copied bit for bit out of the source JSON. The source's own diagnostics
(rho_ci, rho_se, the bootstrap counts, the preflight) are copied too and
describe the SOURCE calibration -- they are not claims about the new
marginal, and the `carried_*` keys say so.

    python scripts/artificial_humans/carry_contribution_copula_params.py \\
        --from <source copula_params.json> \\
        --base <retrained .pt> \\
        --out  <new copula_params.json>
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARRIED = ("rho", "phi", "phi_final", "copula_switch_every")


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = json.loads(args.src.read_text())
    for k in CARRIED:
        assert src.get(k) is not None, f"source JSON carries no {k}"
    assert src.get("phi_kept", True), "source calibration dropped phi"

    out = dict(src)
    out["source_model"] = rel(args.base)
    out["source_model_sha256"] = sha256(args.base)
    out["date"] = datetime.now(timezone.utc).isoformat()
    out["carried_from"] = rel(args.src)
    out["carried_from_source_model"] = src.get("source_model")
    out["carried_from_source_model_sha256"] = src.get("source_model_sha256")
    out["carried_fields"] = list(CARRIED)
    out["carried_note"] = (
        "rho and phi are CARRIED UNCHANGED from the source calibration, not "
        "refit: the copula's parameters are frozen per model family and only "
        "the marginal was retrained. Every other key in this file, including "
        "rho_ci, rho_se, the bootstrap counts, the preflight and the "
        "round-trip bias, describes the SOURCE calibration and is reproduced "
        "here for provenance only -- none of it is a measurement on this "
        "model."
    )
    out["phi_final_reason"] = (
        f"carried unchanged from {rel(args.src)} (phi_final "
        f"{src['phi_final']}); the original ruling is quoted next"
    )
    out["phi_final_reason_source"] = src.get("phi_final_reason")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {rel(args.out)}")
    print(f"  rho={out['rho']} phi_final={out['phi_final']}")
    print(f"  copula_switch_every={out['copula_switch_every']}")
    print(f"  base={out['source_model']}")
    print(f"  sha256={out['source_model_sha256']}")


if __name__ == "__main__":
    main()
