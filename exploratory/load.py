import numpy as np, json, glob
from sklearn.metrics import roc_auc_score, f1_score
R='D:/XrVit/results_s42/'
cal=json.load(open(R+'calibration.json')); thr=json.load(open(R+'thresholds.json'))
def load(tag,split):
    f=glob.glob(R+f'runs/*{tag}*/logits_{split}.npz')[0]; return np.load(f,allow_pickle=True)
def sig(x): return 1/(1+np.exp(-x))
D={}
for split in ['val','test']:
    c=load('CONVNEXT',split); r=load('RAD_DINO',split)
    assert (c['image_index']==r['image_index']).all()
    pc=sig(np.array(cal['cnn']['a'])*c['logits']+np.array(cal['cnn']['b']))
    k=[k for k in cal if k!='cnn'][0]
    pr=sig(np.array(cal[k]['a'])*r['logits']+np.array(cal[k]['b']))
    D[split]=dict(pc=pc,pr=pr,pa=(pc+pr)/2,y=c['labels'].astype(int),pid=c['patient_id'])
T={k:np.array(v['thresholds']) for k,v in thr.items() if v.get('thresholds') is not None}
def macro_auc(p,y): return np.mean([roc_auc_score(y[:,j],p[:,j]) for j in range(14)])
def macro_f1(pred,y): return np.mean([f1_score(y[:,j],pred[:,j],zero_division=0) for j in range(14)])
