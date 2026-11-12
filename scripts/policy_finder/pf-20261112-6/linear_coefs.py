"""Print the features/coefficients of the linear human-behaviour baselines (trained on the human data)."""
import joblib, numpy as np, sys
for f in ['contribution_ridge_best','does_switch_multinomial_best']:
    b=joblib.load(f'artifacts/baselines/{f}.joblib')
    print(f, type(b), list(b.keys()) if isinstance(b,dict) else '')
    m=b.get('model'); feats=b.get('features') or b.get('feature_names')
    print(' features', feats)
    print(' model', m)
    coef=getattr(m,'coef_',None)
    if coef is None and hasattr(m,'steps'): coef=m.steps[-1][1].coef_
    if coef is not None:
        coef=np.atleast_2d(coef)
        for i,fn in enumerate(feats or []):
            print(f'   {fn:40s}', np.round(coef[:,i][-1],4))
    for kk in b:
        if kk not in ('model',): print(' ',kk, str(b[kk])[:300])
