# v3: how does ah's (linear multinomial) punishment depend on the player's previous-round punishment,
# which is the punishment by WHICHEVER manager charged it (so ours, for a player who just joined ah's group)?
import joblib, numpy as np
b=joblib.load('artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib')
e,s=b['estimator'],b['scaler']; L=np.arange(31)
def ep(c,pc,pp,r,valid=1):
    X=np.array([[c,float(c==20),valid,pc,pp,r,float(r==0)]])
    P=e.predict_proba(s.transform(X))[0]
    return (P*e.classes_).sum(), P[0]
for r in [4,8,12,16,20]:
  print('round',r)
  for c in [0,3,6,10,15,20]:
    row=[ep(c,c,pp,r) for pp in [0,2,5,10,15,20,25,30]]
    print(f' c={c:2d} E[p] by prev_p 0,2,5,10,15,20,25,30:',' '.join(f'{m:5.2f}' for m,_ in row),' | P0:',' '.join(f'{z:.2f}' for _,z in row))
# chain: player keeps contributing c, starting prev_p = x at round 4; expected sum of ah's punishments over rounds 4..7 (approx using expected value as next prev_p)
print('chain: total ah punishment rounds 4-7 on a player with constant c, starting prev_p=x (mean-field)')
for c in [0,5,10,15]:
  out=[]
  for x in [0,10,20,30]:
    pp=x; tot=0
    for r in range(4,8):
      m,_=ep(c,c,pp,r); tot+=m; pp=m
    out.append(tot)
  print(f' c={c}', ' '.join(f'{v:6.2f}' for v in out))
