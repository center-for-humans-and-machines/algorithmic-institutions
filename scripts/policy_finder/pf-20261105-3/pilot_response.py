"""Human pilots: response to punishment under random rules vs human managers; persistence."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d['c'] = np.where(d.player_no_input == 1, np.nan, d.contribution)
d = d.sort_values(['experiment_name', 'episode_id', 'player_id', 'round_number'])
key = ['experiment_name', 'episode_id', 'player_id']
g = d.groupby(key)
for k in (1, 2, 3, 5):
    d[f'c{k}'] = g.c.shift(-k)
d['gmean'] = d.groupby(['experiment_name', 'episode_id', 'round_number']).c.transform('mean')
d['others_p'] = (d.groupby(['experiment_name', 'episode_id', 'round_number']).punishment.transform('sum') - d.punishment) / 3
d['nrounds'] = d.groupby(['experiment_name', 'episode_id']).round_number.transform('max') + 1
d['to_end'] = d.nrounds - 1 - d.round_number
print(d.groupby('experiment_name').agg(groups=('episode_id', 'nunique'), c=('c', 'mean'), p=('punishment', 'mean'), payoff=('payoff', 'mean'), timeout=('player_no_input', 'mean')))
print(d.groupby(['experiment_name', 'nrounds', 'round_number']).agg(c=('c', 'mean'), p=('punishment', 'mean')).unstack(0).round(2))
for exp in d.experiment_name.unique():
    s = d[(d.experiment_name == exp) & d.c.notna()]
    print('==', exp)
    for k in (1, 2, 3, 5):
        m = smf.ols(f'c{k} ~ punishment + others_p + c + gmean + to_end', data=s.dropna(subset=[f'c{k}'])).fit()
        print(k, m.params[['punishment', 'others_p', 'c', 'gmean', 'to_end']].round(3).to_dict(), 'se_p', round(m.bse['punishment'], 3))
    s['pb'] = pd.cut(s.punishment, [-1, 0, 3, 6, 10, 31]); s['cb'] = pd.cut(s.c, [-1, 4, 9, 14, 19, 20])
    s['dc'] = s.c1 - s.c
    print(s.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean').round(2))
    print(s.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size'))
