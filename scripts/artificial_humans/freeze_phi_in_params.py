"""Carry the frozen persistence `phi = 1.0` into a freshly refitted params JSON.

`contribution_copula_rho.py` writes the estimated lag-1 ratio as `phi` and
never writes `phi_final`; `make_contribution_copula_artifact.py` stamps
`phi_final` when it is present and the bare `phi` when it is not. A refit of
`rho` alone must therefore re-declare the persistence ruling, or the stamping
would silently replace the shipped `copula_phi = 1.0` with the lag-1 estimate
-- a second, undeclared change to a separately frozen parameter.

This script copies `phi_final` and `phi_final_reason` from the SHIPPED params
JSON into the new one, refusing if the shipped value is anything but 1.0 or if
the new JSON already carries a `phi_final`. Nothing else in the file is
touched; `rho` and every estimate stay exactly as the estimator wrote them.

    python scripts/artificial_humans/freeze_phi_in_params.py \\
        --shipped <old copula_params.json> --params <new copula_params.json>
"""

import argparse
import json
from pathlib import Path

NOTE = (
    "phi is frozen (notes/autoresearch.md section 2): this experiment "
    "declares and refits rho only. phi_final and its reason are carried "
    "over verbatim from the shipped calibration "
    "({shipped}), so the stamped copula_phi is the same 1.0 the baseline "
    "stack runs. The refit's own lag-1 estimate stays in this file as `phi` "
    "and is not stamped."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shipped", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    args = ap.parse_args()

    shipped = json.loads(args.shipped.read_text())
    params = json.loads(args.params.read_text())
    assert shipped.get("phi_final") == 1.0, (
        f"shipped params carry phi_final={shipped.get('phi_final')!r}, "
        f"expected 1.0 -- refusing to guess the frozen persistence"
    )
    assert "phi_final" not in params, "params already carry a phi_final"

    params["phi_final"] = shipped["phi_final"]
    params["phi_final_reason"] = shipped["phi_final_reason"]
    params["phi_frozen_note"] = NOTE.format(shipped=args.shipped)
    args.params.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    print(f"{args.params}: phi_final={params['phi_final']} carried over")
    print(f"  rho (refit, untouched)   {params['rho']}")
    print(f"  phi lag-1 (not stamped)  {params['phi']}")


if __name__ == "__main__":
    main()
