# v7: compact per-game cache of all six sweeps (as v6_cache.py, plus v6), one file at a time (memory-light). Key = seed*1000+slot
# (batch k of a part uses seed part_seed+k, 1000 games per batch). Writes /tmp/v6_<ver>_<part>.npz with
# key, seat, point, punished, lead/po/pt (mean over rounds), members, and int8 C, G, own-P arrays.
import sys, numpy as np, json, gc
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
V={'v1':'pf-20261112-1','v2':'pf-20261112-1-v2','v3':'pf-20261112-1-v3','v4':'pf-20261112-1-v4','v5':'pf-20261112-1-v5','v6':'pf-20261112-1-v6'}
for k,v in V.items():
    sw=json.load(open(B+f'{v}_sweep/sweep.json')); seeds=sw['seeds']
    for pi,part in enumerate('12'):
        d=load(B+f'{v}_sweep/per_round_p{part}of2.parquet')
        n=len(d['C']); key=(seeds[pi]+np.arange(n)//1000)*1000+np.arange(n)%1000
        own=(d['G']==d['mine'][:,None,None])
        pr=1.6*d['C'].astype(np.float32)-d['P']
        po=(pr*own).sum(1); pt=(pr*~own).sum(1)
        Pm=(d['P']*own).astype(np.int8)
        np.savez(f'/tmp/v6_{k}_{part}.npz',key=key,seat=d['mine'].astype(np.int8),point=d['point'],
                 pun=Pm.sum((1,2))>0,po=po.mean(1),pt=pt.mean(1),lead=(po-pt).mean(1),mem=own.sum(1).mean(1),
                 por=po.astype(np.float32),ptr=pt.astype(np.float32),
                 C=d['C'].astype(np.int8),G=d['G'],P=Pm,V=d['V'])
        print(k,part,n,seeds[pi],flush=True); del d,own,pr,po,pt,Pm; gc.collect()
