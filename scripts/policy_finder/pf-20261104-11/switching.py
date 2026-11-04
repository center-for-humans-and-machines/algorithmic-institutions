"""What drives switching? Sims (ah_vs_zero, ah_vs_ah) and human data."""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
pd.set_option('display.width', 250)


def blocks(d, key):
    d = d.sort_values(key + ['round_number'])
    d['blk'] = d.round_number // 4
    gm = d.groupby(key[:-1] + ['round_number', 'group_id']).agg(
        gp=('punishment', 'mean'), gc=('contribution', 'mean'), gpay=('payoff', 'mean')).reset_index()
    d = d.merge(gm, on=key[:-1] + ['round_number', 'group_id'])
    other = gm.copy(); other['group_id'] = 1 - other.group_id
    other = other.rename(columns={'gp': 'op', 'gc': 'oc', 'gpay': 'opay'})
    d = d.merge(other, on=key[:-1] + ['round_number', 'group_id'], how='left')
    # last round of block 0..4 -> next round group
    last = d[d.round_number % 4 == 3].copy()
    nxt = d[(d.round_number % 4 == 0) & (d.round_number > 0)][key + ['round_number', 'group_id']].copy()
    nxt['round_number'] -= 1
    nxt = nxt.rename(columns={'group_id': 'next_g'})
    last = last.merge(nxt, on=key + ['round_number'])
    agg = d[d.round_number < 20].groupby(key + ['blk']).agg(
        bp=('punishment', 'mean'), bc=('contribution', 'mean'), bpay=('payoff', 'mean'),
        bgp=('gp', 'mean'), bgpay=('gpay', 'mean'), bopay=('opay', 'mean')).reset_index()
    last = last.merge(agg, on=key + ['blk'])
    last['sw'] = (last.next_g != last.group_id).astype(float)
    return last


def report(last):
    print('switch rate', last.sw.mean().round(3), 'n', len(last))
    last['ppos'] = pd.cut(last.bp, [-0.1, 0, 1, 3, 6, 10, 31])
    print(last.groupby('ppos').sw.agg(['mean', 'size']).round(3))
    last['payd'] = pd.cut(last.bpay - last.bopay.fillna(last.bpay), [-99, -6, -3, -1, 1, 3, 6, 99])
    print(last.groupby('payd').sw.agg(['mean', 'size']).round(3))
    last['cb'] = pd.cut(last.bc, [-1, 0, 5, 10, 15, 20])
    print(last.groupby('cb').sw.agg(['mean', 'size']).round(3))
    print(pd.crosstab(last.cb, last.ppos, values=last.sw, aggfunc='mean').round(2))


if __name__ == '__main__':
    src = sys.argv[1]
    if src == 'human':
        h = pd.read_csv('experiments/2group_8agent_50ep.csv')
        h = h.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
        last = blocks(h, ['session', 'episode', 'participant_code'])
        report(last)
    else:
        d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
        d = d[d.run.str.contains(src)]
        last = blocks(d, ['episode', 'participant_code'])
        report(last)
