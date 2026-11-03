"""Initial exploration plots for the post-experiment assessment data.

Inputs (relative to repo root):
    data/human_member_assessment.csv  -- Likert ratings (1-5; 0 = no response)
        from group members about Gov A and Gov B, plus self-report items
        (focus, satisfaction_other_strategy, influence_payoff).
    data/assessment_manager.csv  -- free-text descriptions of the strategy
        used by participants who acted as a (human) governor.

Produces (under plots/assessment_plots/):
    member_likert_distributions.png
    member_gov_a_vs_b_means.png
    member_gov_a_vs_b_paired_diff.png
    member_likert_correlation.png
    member_per_session_means.png
    assessment_sample_sizes.png
    manager_response_length.png
    manager_top_words.png

Usage:
    mamba run -n algoinst python scripts/plotting/plot_assessment_data.py

The exact survey question wording is defined in the (external) oTree app and
not present in this repo, so axis labels mirror the column names. Likert
columns range 1-5 with 0 treated as "no response" and dropped per-question.
"""

import argparse
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LIKERT_COLS = [
    "focus",
    "satisfaction_other_strategy",
    "influence_payoff",
    "fairness_gov_a",
    "fairness_gov_b",
    "satisfaction_manager_strategy_gov_a",
    "satisfaction_manager_strategy_gov_b",
    "restrict_freedom_gov_a",
    "restrict_freedom_gov_b",
]

PAIRED_METRICS = [
    ("fairness", "fairness_gov_a", "fairness_gov_b"),
    (
        "satisfaction_manager_strategy",
        "satisfaction_manager_strategy_gov_a",
        "satisfaction_manager_strategy_gov_b",
    ),
    ("restrict_freedom", "restrict_freedom_gov_a", "restrict_freedom_gov_b"),
]

# Common English stopwords; small hand-picked list keeps the script
# dependency-free.
STOPWORDS = {
    "i", "me", "my", "we", "you", "your", "they", "them", "their", "it",
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
    "at", "for", "with", "as", "by", "from", "that", "this", "these",
    "those", "is", "was", "were", "be", "been", "being", "are", "am",
    "do", "did", "does", "doing", "have", "has", "had", "having", "would",
    "could", "should", "will", "shall", "can", "may", "might", "must",
    "so", "also", "than", "then", "there", "here", "what", "which", "who",
    "how", "when", "why", "not", "no", "yes", "all", "any", "some",
    "each", "every", "few", "more", "most", "other", "such", "only",
    "own", "same", "very", "just", "out", "up", "down", "off", "over",
    "under", "again", "into", "about", "after", "before", "between",
    "during", "above", "below", "while", "because", "though", "although",
    "even", "still", "much", "many", "very", "really", "well", "got",
    "get", "make", "made", "see", "saw", "go", "went", "gone", "way",
    "lot", "bit", "thing", "things", "one", "two", "three", "would",
    "wouldnt", "didnt", "dont", "wasnt", "isnt", "ive", "im", "id",
    "thats", "theyre", "theres", "its", "weve", "youre", "doesnt",
    "havent", "couldnt", "wouldn", "didn", "don", "wasn", "isn", "haven",
    "couldn", "give", "gave", "given", "want", "wanted", "felt", "feel",
    "think", "thought", "took", "take", "taken", "taking", "based",
    "rather", "less", "lower", "higher", "amount", "lot", "first",
    "last", "round", "rounds",
}


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _save(fig, output_dir, name):
    out = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _drop_no_response(df):
    """Drop rows where every Likert column is 0 (full skip)."""
    mask = (df[LIKERT_COLS] == 0).all(axis=1)
    if mask.any():
        print(f"Dropping {mask.sum()} all-zero (no-response) rows")
    return df.loc[~mask].copy()


def plot_likert_distributions(member, output_dir):
    """Stacked horizontal bars: share of each Likert level (1-5) per item."""
    counts = []
    for col in LIKERT_COLS:
        s = member[col]
        s = s[s.between(1, 5)]
        c = s.value_counts().reindex(range(1, 6), fill_value=0)
        counts.append(c.rename(col))
    mat = pd.concat(counts, axis=1).T
    pct = mat.div(mat.sum(axis=1), axis=0) * 100

    cmap = sns.color_palette("RdYlBu", n_colors=5)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    left = np.zeros(len(pct))
    for i, level in enumerate(range(1, 6)):
        vals = pct[level].values
        ax.barh(
            pct.index,
            vals,
            left=left,
            color=cmap[i],
            label=f"{level}",
            edgecolor="white",
        )
        for y, (v, l) in enumerate(zip(vals, left)):
            if v >= 5:
                ax.text(
                    l + v / 2,
                    y,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        left += vals

    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Share of responses (%)")
    ax.set_title(
        "Member assessment — Likert distributions (1-5)\n"
        f"n={len(member)} responses; 0-coded skips excluded per item"
    )
    ax.legend(
        title="Likert", bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=False,
    )
    _save(fig, output_dir, "member_likert_distributions.png")


def plot_gov_a_vs_b_means(member, output_dir):
    """Mean +- 95% CI bars for paired Gov-A / Gov-B metrics."""
    rows = []
    for label, col_a, col_b in PAIRED_METRICS:
        for gov, col in [("A", col_a), ("B", col_b)]:
            s = member[col]
            s = s[s.between(1, 5)]
            mean = s.mean()
            sem = s.std(ddof=1) / np.sqrt(len(s))
            rows.append(
                {
                    "metric": label,
                    "governor": gov,
                    "mean": mean,
                    "ci": 1.96 * sem,
                    "n": len(s),
                }
            )
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = {"A": "#3b82f6", "B": "#ef4444"}
    metrics = summary["metric"].unique()
    x = np.arange(len(metrics))
    width = 0.38
    for i, gov in enumerate(["A", "B"]):
        sub = summary[summary["governor"] == gov].set_index("metric").loc[metrics]
        ax.bar(
            x + (i - 0.5) * width,
            sub["mean"],
            width=width,
            yerr=sub["ci"],
            color=palette[gov],
            label=f"Gov {gov}",
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
        for xi, m, n in zip(x, sub["mean"], sub["n"]):
            ax.text(
                xi + (i - 0.5) * width,
                m + 0.05,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_ylim(1, 5)
    ax.set_title("Member ratings of Gov A vs Gov B (mean +/- 95% CI)")
    ax.legend(frameon=False)
    _save(fig, output_dir, "member_gov_a_vs_b_means.png")


def plot_paired_difference(member, output_dir):
    """Within-participant Gov A - Gov B diff per paired metric."""
    fig, axes = plt.subplots(1, len(PAIRED_METRICS), figsize=(11, 4), sharey=True)
    for ax, (label, col_a, col_b) in zip(axes, PAIRED_METRICS):
        sub = member[[col_a, col_b]].copy()
        sub = sub[(sub[col_a].between(1, 5)) & (sub[col_b].between(1, 5))]
        diff = sub[col_a] - sub[col_b]
        sns.histplot(
            diff,
            bins=np.arange(-4.5, 5.5, 1),
            ax=ax,
            color="#6b7280",
            edgecolor="white",
        )
        mean = diff.mean()
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(mean, color="#dc2626", linestyle="-", linewidth=1.5)
        ax.set_title(
            f"{label}\nmean diff={mean:+.2f}  (n={len(diff)})", fontsize=10
        )
        ax.set_xlabel("Gov A - Gov B (per participant)")
        ax.set_xticks(range(-4, 5))
    axes[0].set_ylabel("Participants")
    fig.suptitle(
        "Within-participant rating differences  (negative = Gov B higher)",
        y=1.02,
    )
    _save(fig, output_dir, "member_gov_a_vs_b_paired_diff.png")


def plot_likert_correlation(member, output_dir):
    """Pairwise Spearman correlation across Likert items."""
    sub = member[LIKERT_COLS].replace(0, np.nan)
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title("Likert item Spearman correlations  (0-coded skips dropped)")
    _save(fig, output_dir, "member_likert_correlation.png")


def plot_per_session_means(member, output_dir):
    """Per-session mean Likert response with sample size."""
    sub = member.copy()
    for c in LIKERT_COLS:
        sub[c] = sub[c].where(sub[c].between(1, 5))

    grouped = sub.groupby("session")[LIKERT_COLS].mean()
    sizes = sub.groupby("session").size().rename("n")
    grouped = grouped.join(sizes).sort_values("n", ascending=False)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(grouped))))
    sns.heatmap(
        grouped[LIKERT_COLS],
        cmap="RdYlBu",
        vmin=1,
        vmax=5,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Mean rating"},
        ax=ax,
        yticklabels=[
            f"{s}  (n={n})" for s, n in zip(grouped.index, grouped["n"])
        ],
    )
    ax.set_title("Mean Likert response per session")
    ax.set_xlabel("")
    ax.set_ylabel("session")
    _save(fig, output_dir, "member_per_session_means.png")


def plot_sample_sizes(member, manager, output_dir):
    """Side-by-side bar of responses-per-session for both assessments."""
    m_counts = member.groupby("session").size().rename("members")
    g_counts = manager.groupby("session").size().rename("managers")
    g_filled = (
        manager.assign(filled=manager["manager_behavior"].notna())
        .groupby("session")["filled"]
        .sum()
        .astype(int)
        .rename("managers_with_text")
    )
    merged = pd.concat([m_counts, g_counts, g_filled], axis=1).fillna(0).astype(int)
    merged = merged.sort_values("members", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(merged))))
    y = np.arange(len(merged))
    h = 0.27
    ax.barh(y - h, merged["members"], height=h, label="member responses",
            color="#3b82f6")
    ax.barh(y, merged["managers"], height=h, label="manager rows",
            color="#9ca3af")
    ax.barh(y + h, merged["managers_with_text"], height=h,
            label="manager rows w/ text", color="#10b981")
    ax.set_yticks(y)
    ax.set_yticklabels(merged.index)
    ax.set_xlabel("Count")
    ax.set_title(
        "Assessment sample sizes per session  "
        f"(member n={int(merged['members'].sum())}, "
        f"manager n={int(merged['managers'].sum())}, "
        f"manager w/ text n={int(merged['managers_with_text'].sum())})"
    )
    ax.legend(frameon=False, loc="lower right")
    _save(fig, output_dir, "assessment_sample_sizes.png")


def plot_manager_response_length(manager, output_dir):
    """Histogram of word and character counts in manager_behavior."""
    text = manager["manager_behavior"].dropna().astype(str)
    text = text[text.str.strip().str.len() > 0]
    words = text.str.split().str.len()
    chars = text.str.len()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(words, bins=30, ax=axes[0], color="#2563eb")
    axes[0].axvline(words.median(), color="red", linestyle="--",
                    label=f"median={words.median():.0f}")
    axes[0].set_xlabel("words")
    axes[0].set_ylabel("responses")
    axes[0].set_title(f"Word count  (n={len(words)})")
    axes[0].legend(frameon=False)

    sns.histplot(chars, bins=30, ax=axes[1], color="#16a34a")
    axes[1].axvline(chars.median(), color="red", linestyle="--",
                    label=f"median={chars.median():.0f}")
    axes[1].set_xlabel("characters")
    axes[1].set_ylabel("responses")
    axes[1].set_title(f"Character count  (n={len(chars)})")
    axes[1].legend(frameon=False)

    fig.suptitle("Manager strategy free-text length", y=1.02)
    _save(fig, output_dir, "manager_response_length.png")


def plot_manager_top_words(manager, output_dir, top_n=30):
    """Bar chart of the most frequent (non-stopword) tokens."""
    text = manager["manager_behavior"].dropna().astype(str)
    tokens = []
    for entry in text:
        for tok in re.findall(r"[A-Za-z][A-Za-z']+", entry.lower()):
            if len(tok) <= 2 or tok in STOPWORDS:
                continue
            tokens.append(tok)
    counts = Counter(tokens)
    if not counts:
        print("No tokens to plot for manager_top_words")
        return
    top = counts.most_common(top_n)
    words, freqs = zip(*top)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(words))))
    ax.barh(list(words), list(freqs), color="#7c3aed")
    ax.invert_yaxis()
    ax.set_xlabel("Occurrences")
    ax.set_title(
        f"Top {top_n} non-stopword terms in manager_behavior  "
        f"(n={len(text)} responses, {sum(counts.values())} tokens)"
    )
    _save(fig, output_dir, "manager_top_words.png")


def main():
    parser = argparse.ArgumentParser(
        description="Initial exploration plots for assessment data."
    )
    parser.add_argument(
        "--member-csv",
        default="data/human_member_assessment.csv",
    )
    parser.add_argument(
        "--manager-csv",
        default="data/assessment_manager.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/assessment_plots",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help=(
            "Optional list of session codes to keep. If omitted, all "
            "sessions are used."
        ),
    )
    args = parser.parse_args()

    _ensure_outdir(args.output_dir)

    member = pd.read_csv(args.member_csv)
    manager = pd.read_csv(args.manager_csv)
    print(f"Loaded member assessment: {member.shape}")
    print(f"Loaded manager assessment: {manager.shape}")

    if args.sessions:
        keep = set(args.sessions)
        m_before, g_before = len(member), len(manager)
        member = member[member["session"].isin(keep)].copy()
        manager = manager[manager["session"].isin(keep)].copy()
        print(
            f"Filtered to {len(keep)} sessions: "
            f"member {m_before} -> {len(member)}, "
            f"manager {g_before} -> {len(manager)}"
        )
        missing = keep - set(member["session"]).union(manager["session"])
        if missing:
            print(f"WARNING: no rows for sessions: {sorted(missing)}")

    member = _drop_no_response(member)

    plot_likert_distributions(member, args.output_dir)
    plot_gov_a_vs_b_means(member, args.output_dir)
    plot_paired_difference(member, args.output_dir)
    plot_likert_correlation(member, args.output_dir)
    plot_per_session_means(member, args.output_dir)
    plot_sample_sizes(member, manager, args.output_dir)
    plot_manager_response_length(manager, args.output_dir)
    plot_manager_top_words(manager, args.output_dir)


if __name__ == "__main__":
    main()
