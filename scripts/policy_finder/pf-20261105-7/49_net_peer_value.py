"""v3 real sweep, after the opening, focal player at c<=2: own and peers' rest-of-game change,
fined 20-30 vs unfined, by round; value per event counting own + peers' gains
(0.15 per contribution-round) minus 0.5 x fine. Peers restricted to valid, c>=8 at t, unfined at t."""
import numpy as np, json
a=np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz'); C=a['C'].astype(float);P=a['P'].astype(float);V=a['V'].astype(bool);R=a['R']
pts={p['name'].split('_')[-1]:p['params'] for p in json.load(open('plots/simulation/policy_finder/pf-20261105-7-v3_sweep/sweep.json'))['points']}
W=np.array([pts[r.split('_')[-1]]['window'] for r in R])
for t in range(6,22,2):
    left=23-t
    c=C[:,:,t];p=P[:,:,t]; ok=V[:,:,t:].all(-1)&(W[:,None]<=t)
    rest=C[:,:,t+1:].mean(-1)-c
    # peers: sum over the other 3 players of their rest-of-game change (all valid players)
    vr=V[:,:,t:].all(-1)
    prest=(rest*vr).sum(1,keepdims=True)-rest*vr
    out=[]
    for name,pm in [('unf',p==0),('20-30',p>=20)]:
        m=ok&(c<=2)&pm
        out.append((rest[m].mean(),prest[m].mean(),p[m].mean(),m.sum()))
    (g0,pg0,_,n0),(g1,pg1,f1,n1)=out
    vo=0.15*(g1-g0)*left-0.5*f1; vp=0.15*(pg1-pg0)*left
    print(f't{t:2d} own {g0:+5.1f}->{g1:+5.1f} peers(sum) {pg0:+5.1f}->{pg1:+5.1f}  v_own {vo:+5.1f} v_peers {vp:+5.1f} total {vo+vp:+5.1f}  n {n0} {n1}')
