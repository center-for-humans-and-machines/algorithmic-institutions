"""Refit 79's block-adjusted pooled GP (v3-v6 sweeps + finals) and pickle it with its transform."""
import json, numpy as np, pickle, sys
sys.argv=[sys.argv[0]]
exec(open('scripts/policy_finder/pf-20261105-7/79_pooled_gp_blocks.py').read().split("print(gp.kernel_)")[0])
pickle.dump(dict(gp=gp,mu=mu,sd=sd),open('/tmp/pf7_gp79.pkl','wb'))
print('saved', gp.kernel_)
