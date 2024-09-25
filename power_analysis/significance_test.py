import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm


def check_percent_significant(df: pd.DataFrame, alpha: float = 0.05):
    """Columns should be (experiment, condition, group, participant_id, value)"""

    df = df.copy()
    df["group"] = df["group"] + (df["condition"] * df["group"].max()) + df["condition"]
    successes = 0
    fails = 0
    for experiment_id, dfe in tqdm(
        df.groupby("experiment"),
        total=df["experiment"].nunique(),
        desc="Fitting models",
    ):
        md = smf.mixedlm("value ~ condition", dfe, groups=dfe["group"])
        mdf = md.fit(method=["lbfgs"])
        if mdf.pvalues["condition"] < alpha:
            successes += 1
        else:
            fails += 1

    return successes / (successes + fails)
