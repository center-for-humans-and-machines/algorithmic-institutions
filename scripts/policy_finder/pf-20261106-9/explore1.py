import pandas as pd, numpy as np
df = pd.read_csv('experiments/2group_8agent_50ep.csv')
df = df.drop_duplicates(['session','episode','round_number','participant_code'])
df['game'] = df.session + '_' + df.episode.astype(str)
print('games', df.game.nunique(), 'rows', len(df))
print('mean contribution by round'); print(df.groupby('round_number').contribution.mean().round(1).values)
print('mean punishment by round'); print(df.groupby('round_number').punishment.mean().round(2).values)
# group sizes
gs = df.groupby(['game','round_number','group_id']).size()
print('group size dist', gs.value_counts().sort_index().to_dict())
# per group-round pool
g = df.groupby(['game','round_number','group_id']).agg(cg=('common_good','first'), n=('contribution','size'), c=('contribution','mean'), p=('punishment','mean'))
print('mean pool per group-round', g.cg.mean())
# punishment vs contribution (human managers)
df['cbin'] = pd.cut(df.contribution, [-1,0,5,10,15,19,20])
print(df.groupby('cbin').punishment.agg(['mean','count']))
# relative contribution to group mean
df['gmean'] = df.groupby(['game','round_number','group_id']).contribution.transform('mean')
df['dev'] = df.contribution - df.gmean
df['dbin'] = pd.cut(df.dev, [-21,-10,-5,-2,2,5,21])
print(df.groupby('dbin').punishment.agg(['mean','count']))
df.to_pickle('/tmp/df.pkl')
