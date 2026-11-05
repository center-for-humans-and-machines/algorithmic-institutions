"""Pilot overview: contributions by round and by manager kind, punishment levels, timeouts."""
import pandas as pd, numpy as np
df = pd.read_csv("experiments/pilot_random1_player_round_slim.csv")
print(df.columns.tolist()); print(df.shape)
df["n_rounds"] = df.groupby(["global_group_id","episode"])["round_number"].transform("max")+1
print(df.groupby(["experiment_name","n_rounds"])[["global_group_id"]].nunique())
v = df[df.player_no_input==0]
print("timeout rate", df.groupby("experiment_name").player_no_input.mean())
print(v.groupby(["experiment_name","round_number"])[["contribution","punishment","payoff"]].mean().round(2).unstack(0))
print(v.groupby("experiment_name")[["contribution","punishment","payoff"]].mean())
