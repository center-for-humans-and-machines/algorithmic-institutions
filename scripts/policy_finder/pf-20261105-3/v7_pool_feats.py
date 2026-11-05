"""v7: pool v4, v5, v6 clone sweeps (one structure, three parametrisations) in common
coordinates (shock dose after clamp+floor at c = 0, 5, 10, 13, 16, 18; horizon, lthr,
lstop) and remove the per-seed-slot effect (design index i plays the same games in
every version) by alternating a GP fit and slot means."""
import json, numpy as np, sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

B = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
CS = [0, 5, 10, 13, 16, 18]


def curve(v, p, c):
    c = np.asarray(c, float)
    if v == 'v4':
        fr = np.clip((p['thr'] - c) / p['thr'], 0, 1)
        d = p['pmax'] * fr ** p['gamma']
    elif v == 'v5':
        d = p['pmax'] * (1 - c / 20) ** np.log2(p['pmax'] / p['d10'])
    else:
        g = np.log2(p['pmax'] / p['d10']); k = np.log(p['d10'] / p['d16']) / np.log(2.5)
        d = np.where(c <= 10, p['pmax'] * np.clip(1 - c / 20, 0, 1) ** g,
                     p['d10'] * np.clip((20 - c) / 10, 0, 1) ** k)
    return np.floor(np.minimum(30, d)) * (c < 20)


