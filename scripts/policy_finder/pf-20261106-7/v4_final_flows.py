# v4: in v3's final run (gp point): per opponent and block, my group's size, contributions,
# flows in/out at each decision, newcomers' vs stayers' contributions, the other group's c and p
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv)>1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-7-v3_sweep/final/per_round.parquet'
point = sys.argv[2] if len(sys.argv)>2 else 'pf-20261106-7-v3_gp'
d = pd.read_parquet(f, columns=['run','episode','participant_code','round_number','group_id','contribution','punishment','contribution_valid'])
print(d.run.str.extract(r'managed by (.*)')[0].unique()[:20])
d = d[d.run.str.contains(point+'_vs_|_vs_'+point)].copy()
d['seat'] = np.where(d.run.str.contains(point+'_vs_'), 0, 1)
d['opp'] = d.run.str.extract(r'(opponent_\d+)')[0]
d['mine'] = (d.group_id == d.seat).astype(int)
d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['run','episode','pl','round_number'])
d['prev_mine'] = d.groupby(['run','episode','pl']).mine.shift(1)
d['new'] = (d.mine==1)&(d.prev_mine==0)
d['left'] = (d.mine==0)&(d.prev_mine==1)
d['block'] = d.round_number//4
for o, s in d.groupby('opp'):
    m = s[s.mine==1]; x = s[s.mine==0]
    games = s.groupby(['run','episode']).ngroups
    r = pd.DataFrame({
      'n_mine': m.groupby('block').size()/games/4,
      'c_mine': m.groupby('block').contribution.mean(),
      'p_mine': m.groupby('block').punishment.mean(),
      'c_oth': x.groupby('block').contribution.mean(),
      'p_oth': x.groupby('block').punishment.mean(),
      'in': s[s.new].groupby('block').size()/games,
      'out': s[s.left].groupby('block').size()/games,
    })
    # contributions in the block start round: newcomers vs stayers
    st = s[(s.round_number%4==0)&(s.round_number>0)&(s.mine==1)]
    r['c_new'] = st[st.new].groupby('block').contribution.mean()
    r['c_stay'] = st[~st.new].groupby('block').contribution.mean()
    sx = s[(s.round_number%4==0)&(s.round_number>0)]
    r['c_left'] = sx[sx.left].groupby('block').contribution.mean()
    print(o); print(r.round(2).to_string())
