from load import *
tc=T['CNN alone']; tr=T['RAD-DINO alone']
def run(d,route,keep_pred):
    p=np.where(route[:,None],d['pr'],d['pc'])
    pred=np.where(route[:,None],d['pr']>=tr,keep_pred)
    kept=~route; y=d['y']
    missed=((y==1)&(~pred)&kept[:,None]).sum()
    return route.mean(), macro_auc(p,y), macro_f1(pred,y), missed
rules={
 'current: any p in 0.2-0.8':      lambda pc:(( pc>0.2)&(pc<0.8)).any(1),
 'A: any p < 0.8':                 lambda pc:(pc<0.8).any(1),
 'B: max p < 0.8 (no strong YES)': lambda pc:pc.max(1)<0.8,
 'C: any p in [own thr, 0.8)':     lambda pc:((pc>=tc)&(pc<0.8)).any(1),
}
print(f"{'rule':32s} {'split':5s} {'to RAD-DINO':>11s} {'cost32':>6s} {'AUROC':>7s} {'F1':>7s} {'missed':>7s}")
for n,f in rules.items():
    for s in ['val','test']:
        d=D[s]; r=f(d['pc'])
        keep = d['pc']>=0.8 if n!='C: any p in [own thr, 0.8)' else d['pc']>=0.8
        fr,a,f1,m=run(d,r,keep)
        print(f"{n:32s} {s:5s} {fr:11.1%} {(4.0095+fr*22.849)/22.849:6.2f} {a:7.4f} {f1:7.4f} {m:7d}")
print('ref test: RAD-DINO AUROC .8364 F1 .3788 cost 1.00 | CNN .8263/.3539 cost .18')
