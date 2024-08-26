import pandas as pd


def load_aggregates_df(path) -> pd.DataFrame:
    def clean_run_name(run_name):
        return run_name.replace("ah full managed by ", "")

    df = pd.read_csv(path)
    df["run"] = df["run"].apply(clean_run_name)
    return df
