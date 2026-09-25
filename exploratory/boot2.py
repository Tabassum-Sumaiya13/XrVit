from load import *
tc=T['CNN alone']; tr=T['RAD-DINO alone']; ta=T['Average CNN+RAD-DINO']
def w_avg(d,w=0.4): return w*d['pc']+(1-w)*d['pr']
def exp_err(pc):   # expected number of wrong CNN decisions on this image (uses calibration)
    say_yes=pc>=tc
    return np.where(say_yes,1-pc,pc).sum(1)
def current(pc): return ((pc>0.2)&(pc<0.8)).any(1)
def evaluate(d,route,stage2):
    if stage2=='RD':  p2,pr2=d['pr'],d['pr']>=tr
    else:             p2=w_avg(d); pr2=p2>=ta
    p=np.where(route[:,None],p2,d['pc']); pred=np.where(route[:,None],pr2,d['pc']>=tc)
    y=d['y']; missed=((y==1)&~pred&~route[:,None]).sum()
    return route.mean(),macro_auc(p,y),macro_f1(pred,y),missed
t=D['test']; y=t['y']; tau=0.678
r=exp_err(t['pc'])>tau; p2=w_avg(t)
P={'gate':np.where(r[:,None],p2,t['pc']),'RD':t['pr'],'cur':None}
Pred={'gate':np.where(r[:,None],p2>=ta,t['pc']>=tc),'RD':t['pr']>=tr}
cr=((t['pc']>0.2)&(t['pc']<0.8)).any(1)
P['cur']=np.where(cr[:,None],t['pr'],t['pc']); Pred['cur']=np.where(cr[:,None],t['pr']>=tr,t['pc']>=0.8)
pids=np.unique(t['pid']); idx={p:np.where(t['pid']==p)[0] for p in pids}
rng=np.random.default_rng(20260922); D_={'gate-RD':[], 'gate-cur':[]}
for b in range(300):
    ii=np.concatenate([idx[p] for p in rng.choice(pids,len(pids))])
    m={k:(macro_auc(P[k][ii],y[ii]),macro_f1(Pred[k][ii],y[ii])) for k in P}
    D_['gate-RD'].append(np.subtract(m['gate'],m['RD'])); D_['gate-cur'].append(np.subtract(m['gate'],m['cur']))
for k,v in D_.items():
    v=np.array(v); lo,hi=np.percentile(v,[2.5,97.5],0)
    print(f"{k}: AUROC {v[:,0].mean():+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]   F1 {v[:,1].mean():+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]")
