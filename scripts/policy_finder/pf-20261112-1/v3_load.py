# v3: memory-light loader of a sweep's per-round parquet (rows come in blocks of 24 rounds per player,
# 8 players per episode, in file order): arrays [episodes, 8, 24] plus per-episode point name and our group.
import pyarrow.parquet as pq, numpy as np, re
def load(f):
    t=pq.read_table(f,columns=['punishment','contribution','group_id','contribution_valid','round_number','run'],read_dictionary=['run'])
    rn=t.column('round_number').to_numpy(); assert (rn.reshape(-1,24)==np.arange(24)).all()
    def a(c,dt): return t.column(c).to_numpy().astype(dt).reshape(-1,8,24)
    P=a('punishment',np.int16); C=a('contribution',np.int16); G=a('group_id',np.int8); V=a('contribution_valid',np.int8)
    run=t.column('run').combine_chunks()
    codes=run.indices.to_numpy().reshape(-1,8,24)[:,0,0]; cats=run.dictionary.to_pylist()
    names=[]; mine=[]
    for c in cats:
        m=re.search(r'managed by (\S+)_vs_(\S+)$',c); g0,g1=m.group(1),m.group(2)
        if g0=='ah': names.append(g1); mine.append(1)
        else: names.append(g0); mine.append(0)
    names=np.array(names); mine=np.array(mine)
    return dict(P=P,C=C,G=G,V=V,point=names[codes],mine=mine[codes])
