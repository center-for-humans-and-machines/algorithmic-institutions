"""v3: merge the v1 and v2 sweep caches into /tmp/both.npz (for the surrogate)."""
import numpy as np
a, b = np.load('/tmp/v1sweep.npz'), np.load('/tmp/v2sweep.npz')
np.savez('/tmp/both.npz', C=np.concatenate([a['C'], b['C']]), P=np.concatenate([a['P'], b['P']]),
         V=np.concatenate([a['V'], b['V']]), R=np.concatenate([a['R'], b['R'] + 256]))
