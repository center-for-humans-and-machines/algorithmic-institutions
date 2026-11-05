"""v7: cache v6's sweep as npz (same format as v6_cache)."""
import sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
import importlib.util
spec = importlib.util.spec_from_file_location('c6', 'scripts/policy_finder/pf-20261105-8/v6_cache.py')
src = open('scripts/policy_finder/pf-20261105-8/v6_cache.py').read().split("\ncache('pf")[0]
exec(src)
cache('pf-20261105-8-v6', ('free', 'mid', 'horizon', 'end', 'floor'), '/tmp/v6sweep.npz')
cache('pf-20261105-8-v5', ('standard', 'rate', 'dose', 'horizon', 'end', 'floor'), '/tmp/v5sweep.npz')
