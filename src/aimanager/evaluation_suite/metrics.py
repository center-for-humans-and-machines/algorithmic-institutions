"""Metric extraction for the evaluation suite (#128).

One class per metric group (contribution / switching / punishment), one
method per metric row. Each method turns a canonical agent-round frame
(see convert.py) into the raw material its score is computed from -- no
differencing happens here, so human and simulation extractions can be
inspected side by side. Two kinds of rows:

- distribution: the method returns the observations whose distribution the
  row compares (scored with EMD)
- statistic: the method returns the row's statistic, one value per stratum
  (scored as weighted absolute differences)
- stratified_distribution: the method returns observations with the stratum
  as the index's first level (scored as the weighted mean of per-stratum
  EMDs)

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

RCB_EDGES = [0.0, 0.25, 0.5, 1.0, float("inf")]
RCB_LABELS = ["(0,0.25]", "(0.25,0.5]", "(0.5,1]", ">1"]

RSA_EDGES = [0.0, 3.0, 15.0, float("inf")]
RSA_LABELS = ["1-3", "4-15", "16+"]


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
        conditioned on behaviour (the R rows) carry human-frequency
        weights computed once on the human reference and reused for
        every comparison (#132)."""
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
        kind = self.KINDS[name]
        if kind == "distribution":
            return wasserstein_distance(a, b)
        if weights is None:
            weights = self.weights(name, df_a)
        if kind == "stratified_distribution":
            a_strata = dict(iter(a.groupby(level=0)))
            b_strata = dict(iter(b.groupby(level=0)))
            missing = [
                s for s in weights.index if s not in a_strata or s not in b_strata
            ]
            if missing:
                raise ValueError(
                    f"{name}: empty strata {missing} -- "
                    "no empty-stratum policy exists, see #134"
                )
            emd = pd.Series(
                {
                    s: wasserstein_distance(a_strata[s], b_strata[s])
                    for s in weights.index
                }
            )
            return (emd * weights).sum() / weights.sum()
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


class ResponseMetrics(MetricGroup):
    """R rows (#134): responses conditioned on the stimulus they react to.
    Rows are added step by step; the helpers below are the shared
    derivations. Contribution change dc = c_{t+1} - c_t sits at the
    stimulus round t and is NaN unless both contributions are valid."""

    KINDS = {
        "RCA": "stratified_distribution",
        "RCB": "statistic",
        "RCC": "statistic",
        "RCD": "statistic",
        "RSA": "statistic",
    }

    def rca(self, df):
        """Contribution change by round type: how contributions move after
        the four kinds of round."""
        labelled = self._round_types(self._with_dc(df))
        valid = labelled[labelled["dc"].notna() & labelled["round_type"].notna()]
        return valid.set_index("round_type")["dc"].rename("RCA")

    def rca_weights(self, df):
        """Human frequency of each round type over dc-valid rows."""
        return self.rca(df).groupby(level=0).size()

    def rcb(self, df):
        """Mean contribution change of punished non-full contributors per
        punishment-rate bin; rate = punishment / (20 - contribution),
        punishment per point of shortfall."""
        pop = self._rcb_population(df)
        stat = pop.groupby("rate_bin", observed=False)["dc"].mean()
        stat.index = stat.index.astype(str)
        return stat.rename("RCB")

    def rcb_weights(self, df):
        """Human frequency of each punishment-rate bin."""
        w = self._rcb_population(df).groupby("rate_bin", observed=False).size()
        w.index = w.index.astype(str)
        return w

    def rcc(self, df):
        """Contribution change of full contributors, punished minus
        unpunished -- reaction at the ceiling, where RCB's rate is
        undefined."""
        d = self._with_dc(df)
        full = d[(d["contribution"] == 20) & d["dc"].notna() & d["punishment"].notna()]
        contrast = (
            full.loc[full["punishment"] > 0, "dc"].mean()
            - full.loc[full["punishment"] == 0, "dc"].mean()
        )
        return pd.Series({"contrast": contrast}, name="RCC")

    def rcc_weights(self, df=None):
        return pd.Series({"contrast": 1.0})

    def rcd(self, df):
        """Switching pull: the OLS slope of dc on the gap to the receiving
        group over switch events (C_{n+1} - C_n ~ Chat - C_n) -- how far
        switchers move toward their new group's level."""
        events = self._switch_events(df).dropna(subset=["dc", "receiving_mean"])
        gap = events["receiving_mean"] - events["contribution"]
        slope = gap.cov(events["dc"]) / gap.var()
        return pd.Series({"pull": slope}, name="RCD")

    def rcd_weights(self, df=None):
        return pd.Series({"pull": 1.0})

    def rsa(self, df):
        """Switch share at valid opportunities per received-punishment
        bin, over punished contributors -- who leaves after being
        punished, not how many."""
        pop = self._rsa_population(df)
        stat = pop.groupby("punishment_bin", observed=False)["does_switch"].mean()
        stat.index = stat.index.astype(str)
        return stat.rename("RSA")

    def rsa_weights(self, df):
        """Human frequency of each punishment bin."""
        w = self._rsa_population(df).groupby("punishment_bin", observed=False).size()
        w.index = w.index.astype(str)
        return w

    def _rsa_population(self, df):
        pop = df[df["switch_valid"] & (df["punishment"] > 0)].copy()
        pop["punishment_bin"] = pd.cut(pop["punishment"], RSA_EDGES, labels=RSA_LABELS)
        return pop

    def _rcb_population(self, df):
        d = self._with_dc(df)
        pop = d[
            (d["punishment"] > 0) & (d["contribution"] < 20) & d["dc"].notna()
        ].copy()
        rate = pop["punishment"] / (20 - pop["contribution"])
        pop["rate_bin"] = pd.cut(rate, RCB_EDGES, labels=RCB_LABELS)
        return pop

    def _with_dc(self, df):
        df = df.sort_values(PARTICIPANT + ["round_number"]).copy()
        by_player = df.groupby(PARTICIPANT)
        df["next_contribution"] = by_player["contribution"].shift(-1)
        df["dc"] = df["next_contribution"] - df["contribution"]
        return df

    def _round_types(self, df):
        """Label each agent-round with its RCA round type. Timed-out
        choices at decision rounds are no choice at all and stay
        unlabelled (NA), like every row the taxonomy does not cover."""
        df = df.sort_values(PARTICIPANT + ["round_number"]).copy()
        members = df.groupby(GROUP_CELL)["participant_code"].agg(frozenset)
        now = members.reindex(
            list(zip(df["episode_id"], df["round_number"], df["group_id"]))
        ).to_numpy()
        nxt = members.reindex(
            list(zip(df["episode_id"], df["round_number"] + 1, df["group_id"]))
        ).to_numpy()
        unchanged = pd.Series([a == b for a, b in zip(now, nxt)], index=df.index)

        round_type = pd.Series(pd.NA, index=df.index, dtype="object")
        round_type[~df["switch_mask"]] = "no_switch_allowed"
        round_type[df["switch_valid"] & df["does_switch"]] = "switched"
        stayed = df["switch_valid"] & ~df["does_switch"]
        round_type[stayed & unchanged] = "chose_to_stay"
        round_type[stayed & ~unchanged] = "stayed_comp_changed"
        df["round_type"] = round_type
        return df

    def _switch_events(self, df):
        """One row per switch event, anchored at the decision round n:
        own contribution at n and n+1 (dc), and the receiving group's
        mean contribution at n -- the roster the switcher saw, which
        they are not part of. NaN receiving_mean when the receiving
        group was empty at n."""
        df = self._with_dc(df)
        by_player = df.groupby(PARTICIPANT)
        df["next_group"] = by_player["group_id"].shift(-1)
        events = df[df["does_switch"]].copy()
        means = df.groupby(GROUP_CELL)["contribution"].mean()
        keys = list(
            zip(
                events["episode_id"],
                events["round_number"],
                events["next_group"].astype(int),
            )
        )
        events["receiving_mean"] = means.reindex(keys).to_numpy()
        return events[
            PARTICIPANT
            + [
                "round_number",
                "contribution",
                "next_contribution",
                "dc",
                "receiving_mean",
            ]
        ]


GROUPS = {
    "C": ContributionMetrics(),
    "S": SwitchingMetrics(),
    "P": PunishmentMetrics(),
}
