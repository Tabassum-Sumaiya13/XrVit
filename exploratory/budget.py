from load import *
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
eps=1e-6
def lg(p): p=np.clip(p,eps,1-eps); return np.log(p/(1-p))
def bce(p,y): p=np.clip(p,eps,1-eps); return -(y*np.log(p)+(1-y)*np.log(1-p))
tc=T['CNN alone']; ta=T['Average CNN+RAD-DINO']
def feats(pc):
    L=lg(pc); return np.hstack([L, pc*(1-pc), np.abs(L-lg(tc)), (-(pc*np.log(pc+eps)+(1-pc)*np.log(1-pc+eps))).sum(1,keepdims=True)])
def scores(d, model=None):
    pc=d['pc']; S={}
    S['current(max unsure)']=np.minimum(pc,1-pc).max(1)
    S['sum entropy']=(-(pc*np.log(pc+eps)+(1-pc)*np.log(1-pc+eps))).sum(1)
    S['near own threshold']=np.exp(-np.abs(lg(pc)-lg(tc))).sum(1)
    if model is not None: S['learned gate']=model.predict(feats(pc))
    S['ORACLE (uses labels)']=(bce(pc,d['y'])-bce(d['pa'],d['y'])).sum(1)
    return S
v,t=D['val'],D['test']
target=(bce(v['pc'],v['y'])-bce(v['pa'],v['y'])).sum(1)
m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,random_state=42).fit(feats(v['pc']),target)
# cross-fitted val preds for val-quantile thresholds
cv=np.zeros(len(target))
for tr,te in GroupKFold(5).split(target,groups=v['pid']):
    cv[te]=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,random_state=42).fit(feats(v['pc'][tr]),target[tr]).predict(feats(v['pc'][te]))
Sv=scores(v,m); Sv['learned gate']=cv
St=scores(t,m)
def mix(d,route):
    p=np.where(route[:,None],d['pa'],d['pc']); pred=np.where(route[:,None],d['pa']>=ta,d['pc']>=tc)
    return macro_auc(p,d['y']),macro_f1(pred,d['y'])
print('ref test: CNN .8263/.3539  RAD-DINO .8364/.3788  AVG .8428/.3818')
print(f"{'gate':24s} {'B':>4s} {'valB->testB':>11s} {'AUROC':>7s} {'F1':>7s} {'cost32':>6s} {'cost1':>6s}")
for name in St:
    for B in [0.1,0.2,0.3,0.4,0.5,0.6]:
        qv=np.quantile(Sv[name],1-B); realized=(St[name]>qv).mean()
        route=St[name]>np.quantile(St[name],1-B)   # budget enforced on test batch
        a,f=mix(t,route)
        print(f"{name:24s} {B:4.0%} {realized:11.0%} {a:7.4f} {f:7.4f} {(4.0095+B*22.849)/22.849:6.2f} {(11.37+B*22.17)/22.17:6.2f}")
    print()
