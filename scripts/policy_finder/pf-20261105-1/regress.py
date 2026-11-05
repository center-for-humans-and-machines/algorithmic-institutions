import pandas as pd, numpy as np
def prep_sim():
    d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
    d = d[d.run.str.endswith('humanlike')].copy()
    d['valid'] = d.contribution_valid.astype(int); d.loc[d.valid==0,'contribution']=0
    d['ep']=d.episode; d['pc']=d.participant_code; return d
def prep_pilot(exp):
    d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
    d = d[d.experiment_name==exp].copy(); d['valid']=1-d.player_no_input
    d['ep']=d.episode_id; d['pc']=d.participant_code; return d
def fit(d, name):
    d = d.sort_values(['ep','pc','round_number'])
    s = d[d.valid==1].groupby(['ep','round_number']).contribution.agg(['sum','count'])
    d = d.join(s, on=['ep','round_number'])
    d['oth'] = (d['sum']-d.contribution*d.valid)/(d['count']-d.valid).clip(lower=1)
    g = d.groupby(['ep','pc'])
    print('\n==',name)
    for k in [1,3,5]:
        d['cn']=g.contribution.shift(-k); d['vn']=g.valid.shift(-k)
        x = d[(d.valid==1)&(d.vn==1)]
        for lab, m in [('c<10', x.contribution<10), ('c>=10', x.contribution>=10)]:
            y = x[m]
            X = np.c_[np.ones(len(y)), y.contribution, y.oth, y.punishment, (y.punishment>0), y.round_number]
            b = np.linalg.lstsq(X, y.cn, rcond=None)[0]
            print(f' +{k} {lab:6s} n={len(y):6d} const={b[0]:.2f} c={b[1]:.2f} oth={b[2]:.2f} p={b[3]:.3f} p>0={b[4]:.2f} rnd={b[5]:.3f}')
fit(prep_pilot('random_1'),'pilot random_1')
fit(prep_pilot('trail_rounds_2'),'pilot human mgr')
fit(prep_sim(),'sim humanlike')
