"""Metric extraction for the evaluation suite (#128).

One class per metric group (contribution / switching / punishment), one
method per metric row. Each method turns a canonical agent-round frame
(see convert.py) into the raw material its score is computed from -- no
differencing happens here, so human and simulation extractions can be
inspected side by side. Two kinds of rows:

- distribution: the method returns the observations whose distribution the
  row compares (scored with EMD in a later step)
- statistic: the method returns the row's statistic, one value per stratum
  (scored as weighted absolute differences in a later step)

Empty groups produce no rows in the canonical frame, so group-level
extractions lose exactly the empty cell (CC) or the whole game-round where
the difference needs both groups (CE); participant-level metrics keep the
surviving group's rows. NaN (no-input) values drop out of every mean and
count. See notes/evaluation_metric_defs.md for the row definitions.
"""

import pandas as pd
from scipy.stats import wasserstein_distance

PARTICIPANT = ["episode_id", "participant_code"]
GROUP_CELL = ["episode_id", "round_number", "group_id"]

ROUNDS = pd.RangeIndex(24, name="round_number")
DECISION_ROUNDS = pd.Index([3, 7, 11, 15, 19], name="round_number")


def _uniform(index):
    return pd.Series(1.0, index=index)


class MetricGroup:
    """Subclasses define KINDS ({row name: 'distribution' | 'statistic'})
    and one extraction method per row, named after the row in lowercase.
    Every method returns a named pd.Series -- observations for
    distributions, per-stratum values for statistics. Statistic rows also
    define a <row>_weights method returning the number of observations
    underlying each stratum's value."""

    KINDS = {}

    def extract_all(self, df):
        return {name: getattr(self, name.lower())(df) for name in self.KINDS}

    def weights(self, name, df):
        """Stratum weights for a statistic row. Strata fixed by the game
        design (rounds, boundary cells, switching opportunities -- all of
        C/S/P) carry uniform precomputed weights and ignore df; strata
        conditioned on behaviour (the R rows, later branch) carry
        human-frequency weights computed once on the human reference and
        reused for every comparison (#132)."""
        return getattr(self, f"{name.lower()}_weights")(df)

    def d(self, name, df_a, df_b, weights=None):
        """Discrepancy between two datasets on one row, callable on any
        episode subset (#132's scoring pipeline consumes this directly).
        Distribution rows: EMD (1-Wasserstein on the empirical samples,
        no binning). Statistic rows: weighted mean absolute per-stratum
        difference under the row's weight scheme (see weights()); pass
        precomputed weights or let them default to weights(name, df_a).
        Every weighted stratum must be present on both sides -- an empty
        stratum raises, and a policy only gets designed if a real
        candidate model ever triggers this (#134)."""
        extract = getattr(self, name.lower())
        a, b = extract(df_a), extract(df_b)
        if self.KINDS[name] == "distribution":
            return wasserstein_distance(a, b)
        if weights is None:
            weights = self.weights(name, df_a)
        aligned = pd.concat({"a": a, "b": b, "w": weights}, axis=1)
        empty = aligned["a"].isna() | aligned["b"].isna()
        if empty.any():
            raise ValueError(
                f"{name}: empty strata {aligned.index[empty].tolist()} -- "
                "no empty-stratum policy exists, see #134"
            )
        return ((aligned["a"] - aligned["b"]).abs() * aligned["w"]).sum() / aligned[
            "w"
        ].sum()

    def std_diff(self, name, df, reference_df):
        """Signed std difference in raw units (df minus reference) -- the
        retained diagnostic for CA/CC/CE, reported unnormalised."""
        extract = getattr(self, name.lower())
        return extract(df).std() - extract(reference_df).std()


class ContributionMetrics(MetricGroup):
    KINDS = {
        "CA": "distribution",
        "CB": "statistic",
        "CC": "distribution",
        "CD": "distribution",
        "CE": "distribution",
        "CF": "statistic",
    }

    def ca(self, df):
        """Participant mean contributions."""
        obs = df.groupby(PARTICIPANT)["contribution"].mean().dropna()
        return obs.rename("CA")

    def cb(self, df):
        """Round mean contributions."""
        stat = df.groupby("round_number")["contribution"].mean()
        return stat.rename("CB")

    def cc(self, df):
        """Group mean contributions per (game, round)."""
        obs = df.groupby(GROUP_CELL)["contribution"].mean().dropna()
        return obs.rename("CC")

    def cd(self, df):
        """Raw contributions."""
        return df["contribution"].dropna().rename("CD")

    def ce(self, df):
        """Signed group contribution differences (group 0 minus group 1)
        per (game, round); drops game-rounds where either group is empty."""
        means = df.groupby(GROUP_CELL)["contribution"].mean()
        wide = means.unstack("group_id").dropna()
        return (wide[0] - wide[1]).rename("CE")

    def cf(self, df):
        """Share of contributions at the boundaries (0 and 20) per round."""
        valid = df.dropna(subset=["contribution"])
        shares = valid.groupby("round_number")["contribution"].agg(
            share_at_0=lambda c: c.eq(0).mean(),
            share_at_20=lambda c: c.eq(20).mean(),
        )
        return shares.stack().rename("CF")

    def cb_weights(self, df=None):
        return _uniform(ROUNDS)

    def cf_weights(self, df=None):
        cells = pd.MultiIndex.from_product([ROUNDS, ["share_at_0", "share_at_20"]])
        return _uniform(cells)


class SwitchingMetrics(MetricGroup):
    KINDS = {
        "SA": "statistic",
        "SB": "statistic",
        "SC": "distribution",
    }

    def sa(self, df):
        """Overall switch rate over valid switching opportunities."""
        valid = df[df["switch_valid"]]
        rate = valid["does_switch"].mean()
        return pd.Series({"switch_rate": rate}, name="SA")

    def sb(self, df):
        """Switch rate per switching opportunity (decision round)."""
        valid = df[df["switch_valid"]]
        stat = valid.groupby("round_number")["does_switch"].mean()
        return stat.rename("SB")

    def sc(self, df):
        """Size of the larger group per (game, round), rounds 4 onward.
        Empty-group rounds are kept: larger-group size 8 is the maximal
        segregation observation this row exists to measure."""
        sizes = df[df["round_number"] >= 4].groupby(GROUP_CELL).size()
        obs = sizes.groupby(["episode_id", "round_number"]).max()
        return obs.rename("SC")

    def sa_weights(self, df=None):
        return pd.Series({"switch_rate": 1.0})

    def sb_weights(self, df=None):
        return _uniform(DECISION_ROUNDS)


class PunishmentMetrics(MetricGroup):
    KINDS = {
        "PA": "distribution",
        "PB": "statistic",
        "PC": "statistic",
    }

    def pa(self, df):
        """Raw received punishments, zeros included."""
        return df["punishment"].dropna().rename("PA")

    def pb(self, df):
        """Round mean punishments."""
        stat = df.groupby("round_number")["punishment"].mean()
        return stat.rename("PB")

    def pc(self, df):
        """Share of players receiving zero punishment per round (the
        extensive margin: whether to punish, as against how much)."""
        valid = df.dropna(subset=["punishment"])
        stat = valid.groupby("round_number")["punishment"].agg(lambda p: p.eq(0).mean())
        return stat.rename("PC")

    def pb_weights(self, df=None):
        return _uniform(ROUNDS)

    pc_weights = pb_weights


GROUPS = {
    "C": ContributionMetrics(),
    "S": SwitchingMetrics(),
    "P": PunishmentMetrics(),
}
