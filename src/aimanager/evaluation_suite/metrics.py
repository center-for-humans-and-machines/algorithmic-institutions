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

PARTICIPANT = ["episode_id", "participant_code"]
GROUP_CELL = ["episode_id", "round_number", "group_id"]


class MetricGroup:
    """Subclasses define KINDS ({row name: 'distribution' | 'statistic'})
    and one extraction method per row, named after the row in lowercase.
    Every method returns a named pd.Series -- observations for
    distributions, per-stratum values for statistics."""

    KINDS = {}

    def extract_all(self, df):
        return {name: getattr(self, name.lower())(df) for name in self.KINDS}


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


GROUPS = {"C": ContributionMetrics()}
