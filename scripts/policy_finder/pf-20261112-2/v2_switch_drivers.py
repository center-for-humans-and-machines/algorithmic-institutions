"""v2: in v1's sweep, what drives leaving my group / joining it at a decision round:
import warnings; warnings.filterwarnings("ignore")
punishment received at lags 0..3 before the decision, own contribution, and the
per-capita common-good gap (mine - ah's) at the decision round and earlier."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq, pyarrow.compute as pc
base = 'plots/simulation/policy_finder/pf-20261112-2_sweep/'
out = []
for f in sorted(glob.glob(base + 'per_round_p*.parquet')):
    runs_all = pq.read_table(f, columns=['run'], read_dictionary=['run']).column('run').unique().dictionary.to_pylist() if False else sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
    for k0 in range(0, len(runs_all), 32):
        t = pq.read_table(f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'common_good', 'contribution', 'contribution_valid', 'group_id', 'run'], read_dictionary=['run', 'participant_code'], filters=[('run', 'in', runs_all[k0:k0 + 32])])
        d = t.to_pandas()
        cats = pd.Series(d.run.cat.categories)
        first = cats.str.extract(r'by (\S+)_vs_')[0].str.startswith('pf').values
        code = d.run.cat.codes.values.astype(np.int64)
        d['myg'] = np.where(first[code], 0, 1)
        d['ep'] = code * 10000 + d.episode.values
        pcat = pd.Series(d.participant_code.cat.categories).str.split('_').str[0].astype(int).values
        d['pid'] = pcat[d.participant_code.cat.codes.values.astype(np.int64)]
        d = d.drop(columns=['participant_code', 'run'])
        d = d.sort_values(['ep', 'pid', 'round_number'])
        n_pl = d.groupby('ep').pid.nunique().iloc[0]
        E = d.ep.nunique()
        def A(col): return d[col].values.reshape(E, -1, 24)
        P, CG, C, V, G = A('punishment'), A('common_good'), A('contribution'), A('contribution_valid'), A('group_id')
        MY = A('myg')[:, 0, 0]
        mine = (G == MY[:, None, None])
        # per-capita common good of my group and ah's per round (from the per-player column)
        cg_me = np.where(mine, CG, np.nan); cg_ah = np.where(~mine, CG, np.nan)
        with np.errstate(all='ignore'):
            cgm = np.nanmax(cg_me, 1); cga = np.nanmax(cg_ah, 1)
            cbar = np.nanmean(np.where(mine, C, np.nan), 1)
        for dr in [3, 7, 11, 15, 19]:
            sw = (G[:, :, dr + 1] != G[:, :, dr])
            for who, msk in [('leave_mine', mine[:, :, dr]), ('leave_ah', ~mine[:, :, dr])]:
                e, i = np.nonzero(msk & V[:, :, dr])
                rec = pd.DataFrame({'who': who, 'dr': dr, 'sw': sw[e, i].astype(float), 'c': C[e, i, dr],
                                    'cblk': C[e, i, dr - 3:dr + 1].mean(1),
                                    'gap': cgm[e, dr] - cga[e, dr], 'gap_blk': np.nanmean(cgm[e, dr - 3:dr + 1] - cga[e, dr - 3:dr + 1], 1),
                                    'nme': mine[e, :, dr].sum(1)})
                for L in range(4):
                    rec[f'p{L}'] = P[e, i, dr - L]
                out.append(rec)
x = pd.concat(out)
x.to_parquet('/tmp/v2_switch.parquet')
