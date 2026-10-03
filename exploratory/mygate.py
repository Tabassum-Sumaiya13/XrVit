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
v,t=D['val'],D['test']
rd_val_f1=macro_f1(v['pr']>=tr,v['y'])
ev,et=exp_err(v['pc']),exp_err(t['pc'])
print(f"RAD-DINO val F1 {rd_val_f1:.4f}; rule: val F1 >= {rd_val_f1-0.01:.4f}, fewest routed\n")
for stage2 in ['RD','AVG']:
    best=None
    for q in np.linspace(0.05,0.95,91):         # candidate tau = val quantiles
        tau=np.quantile(ev,q); fr,a,f,m=evaluate(v,ev>tau,stage2)
        if f>=rd_val_f1-0.01 and (best is None or fr<best[1]): best=(tau,fr,f)
    tau=best[0]
    fv=evaluate(v,ev>tau,stage2); ft=evaluate(t,et>tau,stage2)
    print(f"expected-error gate, stage2={stage2}: tau={tau:.3f}")
    for nm,r in [('val',fv),('test',ft)]:
        print(f"   {nm:4s} routed {r[0]:6.1%} cost {(4.0095+r[0]*22.849)/22.849:.2f}  AUROC {r[1]:.4f}  F1 {r[2]:.4f}  missed {r[3]}")
print("\ncurrent gate (0.2-0.8), stage2=RD:")
for nm,d in [('val',v),('test',t)]:
    r=evaluate(d,current(d['pc']),'RD'); print(f"   {nm:4s} routed {r[0]:6.1%} cost {(4.0095+r[0]*22.849)/22.849:.2f}  AUROC {r[1]:.4f}  F1 {r[2]:.4f}  missed {r[3]}")
print("\nSame budget on test (top-k% per batch), stage2=AVG:")
rng=np.random.default_rng(0)
for B in [0.3,0.5,0.7]:
    a=evaluate(t,et>np.quantile(et,1-B),'AVG'); c=evaluate(t,np.minimum(t['pc'],1-t['pc']).max(1)>np.quantile(np.minimum(t['pc'],1-t['pc']).max(1),1-B),'AVG')
    rr=np.mean([evaluate(t,rng.random(len(et))<B,'AVG')[1:3] for _ in range(5)],0)
    print(f"   B={B:.0%} cost {(4.0095+B*22.849)/22.849:.2f} | exp-error {a[1]:.4f}/{a[2]:.4f} | current-style {c[1]:.4f}/{c[2]:.4f} | random {rr[0]:.4f}/{rr[1]:.4f}")
print("\nref test: RAD-DINO .8364/.3788 cost 1.00 | 0.4/0.6 avg .8430/.3825 cost 1.18 | CNN .8263/.3539 cost .18")
