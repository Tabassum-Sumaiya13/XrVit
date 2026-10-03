from load import *
exec(open('budget.py').read().split("print('ref test")[0])
from sklearn.metrics import f1_score
def best_thr(p,y):
    ts=np.linspace(0.02,0.6,59); return np.array([ts[np.argmax([f1_score(y[:,j],p[:,j]>=u,zero_division=0) for u in ts])] for j in range(14)])
# 1) fusion weight chosen on val
print('fusion weight w*CNN+(1-w)*RD  (val AUROC | test AUROC, test F1 w/ val-tuned thr)')
for w in [0.2,0.3,0.4,0.5,0.6]:
    pv=w*v['pc']+(1-w)*v['pr']; pt=w*t['pc']+(1-w)*t['pr']; th=best_thr(pv,v['y'])
    print(f"  w={w}: val {macro_auc(pv,v['y']):.4f} | test {macro_auc(pt,t['y']):.4f}  F1 {macro_f1(pt>=th,t['y']):.4f}")
# 2) learned gate, large budgets, second stage = avg vs RD
print('\nlearned gate, budget enforced per batch')
for B in [0.5,0.6,0.7,0.8,0.9]:
    r=St['learned gate']>np.quantile(St['learned gate'],1-B)
    a1,f1=mix(t,r)
    p2=np.where(r[:,None],t['pr'],t['pc']); a2=macro_auc(p2,t['y']); f2=macro_f1(np.where(r[:,None],t['pr']>=T['RAD-DINO alone'],t['pc']>=tc),t['y'])
    print(f"  B={B:.0%}: stage2=AVG {a1:.4f}/{f1:.4f}  stage2=RD {a2:.4f}/{f2:.4f}  cost32 {(4.0095+B*22.849)/22.849:.2f}")
# 3) how much do the gates know? AUROC of gate score for predicting 'avg helps a lot' (top-30% oracle benefit)
ob=St['ORACLE (uses labels)']; tgt=ob>np.quantile(ob,0.7)
print('\nHow well each gate finds the images where stage 2 helps most (AUROC, 0.5=random):')
for k in St:
    if 'ORACLE' not in k: print(f"  {k:22s} {roc_auc_score(tgt,St[k]):.3f}")
# 4) where is the oracle benefit? positives vs negatives
y=t['y']; print('\nOracle top-30% images: mean findings', y[tgt].sum(1).mean().round(2), 'vs rest', y[~tgt].sum(1).mean().round(2))
print('share of oracle benefit from positive labels:', ((bce(t['pc'],y)-bce(t['pa'],y))*y).sum()/ (bce(t['pc'],y)-bce(t['pa'],y)).sum())
