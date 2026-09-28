"""Seed-ensemble artificial humans: one member per episode.

PR #140's Option 1 as a noise model: instead of a shared per-cell latent
(the herding copula, `generic/copula.py`), the correlated component of the
group's error comes from drawing ONE ensemble member (one training seed) at
the start of every episode and letting it decide every agent-round of that
episode independently. The shared error is then the members' disagreement,
not a fitted rho. A sim config points `contribution_model` at a
`*.ensemble.yml` listing the member artifacts (paths relative to the sim's
basedir); `load_ah_model` dispatches on the extension.
"""

import os
import random

import yaml


class SeedEnsembleAH:
    def __init__(self, members):
        assert members, "an ensemble needs at least one member"
        y = {m.y_name for m in members}
        assert len(y) == 1, f"members predict different targets: {y}"
        assert all(m.copula_rho == 0.0 for m in members), (
            "ensemble members must be bare trunks (copula_rho == 0): the "
            "ensemble draw IS the shared latent"
        )
        self.members = members
        self.default_values = members[0].default_values
        self.device = members[0].device
        self.current = None

    @classmethod
    def load(cls, path, device=None, basedir="."):
        from aimanager.artificial_humans import GraphNetwork

        with open(path) as fh:
            spec = yaml.safe_load(fh)
        members = [
            GraphNetwork.load(os.path.join(basedir, p), device=device)
            for p in spec["members"]
        ]
        return cls(members)

    def predict(self, data, sample=True, reset_rnn=True, edge_index=None):
        # reset_rnn is true exactly at round 0, i.e. once per episode: that is
        # the draw. Python's global RNG, seeded by the simulation's `seed`.
        if bool(reset_rnn) or self.current is None:
            self.current = self.members[random.randrange(len(self.members))]
        return self.current.predict(
            data, sample=sample, reset_rnn=reset_rnn, edge_index=edge_index
        )
