import numpy as np, torch as th, sys
sys.path.insert(0,'scripts/policy_finder/pf-20261105-7')
from sur2 import Net, simulate
th.set_num_threads(2)
net=Net(); net.load_state_dict(th.load('scripts/policy_finder/pf-20261105-7/sur2.pt',weights_only=True))
_C0=th.tensor(np.load('/tmp/pf7_arr.npz')['C'][:,:,0])
def run(rule, G=4000, seed=0, verbose=False, label=''):
    g=th.Generator().manual_seed(seed)
    C0=_C0[th.randint(0,_C0.shape[0],(G,),generator=g)]
    pay,C,P=simulate(net,rule,C0,seed=seed)
    m=pay.mean().item(); se=pay.std().item()/G**.5
    if verbose:
        print(f'{label:40s} {m:6.1f} +- {se:.1f}  c:', ' '.join(f'{x:.0f}' for x in C.mean((0,1)).numpy()[::3]), ' p:', ' '.join(f'{x:.1f}' for x in P.mean((0,1)).numpy()[::3]), flush=True)
    return m
