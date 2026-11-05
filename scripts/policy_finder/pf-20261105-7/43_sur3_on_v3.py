"""Out-of-sample check: sur3_all (trained on v1+v2 real games) replays v3's 256 design points;
compare with v3's real sweep scores (correlation, offset, top-10 overlap)."""
import torch as th, numpy as np, sys, yaml, json
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7'); sys.path.insert(0,'src')
from aimanager.manager.rule import sobol_design
from sur3 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur3_all.pt',weights_only=True)); net.eval()
C0all=th.tensor(np.load('/tmp/pf7/pf-20261105-7-v2_sweep.npz')['C'][:,:,0]).float()
G=300
pts=sobol_design(yaml.safe_load(open('configs/managers/rule_based/pf-20261105-7-v3.yml')),256)
keys=('free_rider_fine','curvature','window','net_floor','net_end','net_scale')
def v3(par):
    F,gm,w,fl,E,sc=[par[:,i:i+1] for i in range(6)]
    def r(C,P,t):
        c=C[...,-1]; f=F*((20-c).clamp(min=0)/20)**gm
        return f*((t<w).float()+sc*((t>=w)&(t<E)).float()*(c<=fl).float())
    return r
s=np.zeros(256)
for i0 in range(0,256,64):
    par=th.tensor([[p[k] for k in keys] for p in pts[i0:i0+64]],dtype=th.float32).repeat_interleave(G,0)
    C0=C0all[th.randint(0,C0all.shape[0],(par.shape[0],),generator=th.Generator().manual_seed(i0))]
    s[i0:i0+64]=simulate(net,v3(par),C0,seed=i0).reshape(64,G).mean(1).numpy()
real=np.array([p['pool'] for p in json.load(open('plots/simulation/policy_finder/pf-20261105-7-v3_sweep/sweep.json'))['points']])
np.save('/tmp/pf7/sur3_on_v3.npy',s)
print('corr',np.corrcoef(s,real)[0,1].round(3),'mean sur',s.mean().round(1),'real',real.mean().round(1))
from scipy.stats import spearmanr; print('spearman',spearmanr(s,real)[0].round(3))
ts=set(np.argsort(-s)[:20]); tr=set(np.argsort(-real)[:20]); print('top20 overlap',len(ts&tr))
print('real score of surrogate top10',real[np.argsort(-s)[:10]].round(1))
