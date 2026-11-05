import pandas as pd, numpy as np
def load():
    df = pd.read_csv('/opt/algorithmic-institutions/experiments/2group_8agent_50ep.csv')
    df = df.drop_duplicates(['session','episode','round_number','participant_code']).copy()
    df['game'] = df['session'] + '_' + df['episode'].astype(str)
    df = df.sort_values(['game','player_id','round_number'])
    g = df.groupby(['game','player_id'])
    df['next_c'] = g['contribution'].shift(-1)
    df['next_group'] = g['group_id'].shift(-1)
    df['next_valid'] = 1 - g['player_no_input'].shift(-1)
    df['prev_c'] = g['contribution'].shift(1)
    df['prev_p'] = g['punishment'].shift(1)
    # group stats
    v = df[df.player_no_input == 0]
    gs = df.groupby(['game','round_number','group_id']).agg(n=('player_id','size'), cg=('common_good','first'),
        mc=('contribution','mean'), mp=('punishment','mean')).reset_index()
    df = df.merge(gs, on=['game','round_number','group_id'])
    other = gs.copy(); other['group_id'] = 1 - other['group_id']
    other = other.rename(columns={'n':'n_o','cg':'cg_o','mc':'mc_o','mp':'mp_o'})
    df = df.merge(other, on=['game','round_number','group_id'], how='left')
    df['n_o'] = df['n_o'].fillna(0)
    # leave-one-out group mean contribution
    df['loo_mc'] = np.where(df.n>1, (df.mc*df.n - df.contribution)/(df.n-1).clip(lower=1), np.nan)
    return df, gs
