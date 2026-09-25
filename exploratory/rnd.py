from load import *
exec(open('budget.py').read().split("print('ref test")[0])
rng=np.random.default_rng(42)
for B in [0.3,0.5,0.8]:
    rs=[mix(t,rng.random(len(t['y']))<B) for _ in range(5)]
    a,f=np.mean(rs,0)
    e=St['sum entropy']; ae,fe=mix(t,e>np.quantile(e,1-B))
    print(f"B={B:.0%}: random {a:.4f}/{f:.4f}   entropy gate {ae:.4f}/{fe:.4f}")
# RAD-DINO overfit check not possible (no train logits); CNN missed positives analysis
y=t['y']; miss=(y==1)&(t['pc']<tc); print('CNN-missed positives:',miss.sum(),' of which RAD-DINO finds:',(miss&(t['pr']>=T['RAD-DINO alone'])).sum())
