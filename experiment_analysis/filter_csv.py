import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--session_ids", type=str, required=True, nargs="+")

    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    print(args.session_ids, df["session"].unique())
    df = df[df["session"].isin(args.session_ids)]
    df.to_csv(args.output_file, index=False)
