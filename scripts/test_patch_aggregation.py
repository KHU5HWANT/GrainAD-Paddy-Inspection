"""Compare patch aggregation with unchanged weights and a fixed retrospective split."""
import sys,os,csv,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME',str(ROOT/'datasets/torch-cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/matplotlib'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from data.grain import GrainDataset,initialize_worker
from models.simplenet import SimpleNet
from models.backbones import load
from utils.calibration import recall_threshold
OUT=ROOT/'results/paddy_full/aggregation_experiment'
FRACTIONS=[.001,.005,.01,.02,.05,.10]
def metrics(y,s,t):
 p=s>=t
 return dict(auroc=float(roc_auc_score(y,s)),accuracy=float(accuracy_score(y,p)),recall=float(recall_score(y,p)),precision=float(precision_score(y,p)),f1=float(f1_score(y,p)),false_positive_rate=float(p[y==0].mean()),detected=int(p[y==1].sum()),false_alarms=int(p[y==0].sum()))
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
 manifest={Path(r['image']).name:r for r in csv.DictReader((ROOT/'results/paddy_full/calibration/split_manifest.csv').open())}
 assert len(manifest)==4000
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('Device:',device,flush=True)
 ds=GrainDataset(str(ROOT/'datasets/grain'),classname='paddy',is_train=False,five_split=True,with_mask=False,resize=[224,224],imagesize=[224,224],padding=0,anomaly_source_path='',generate_anomalies=False)
 loader=DataLoader(ds,batch_size=4,num_workers=2,worker_init_fn=initialize_worker)
 cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/models/simplenet.yaml'));cfg.update(device=device,input_shape=[3,224,224],ica_path=str(ROOT/'results/paddy_full/ica/models/ica.pth'),visualization=False)
 model=SimpleNet(device);model.load(backbone=load('wideresnet50'),**cfg)
 ck=ROOT/'results/paddy_full/grainad/models/grain/ckpt.pth';state=torch.load(ck,map_location=device,weights_only=True)
 model.discriminator.load_state_dict(state['discriminator']);model.pre_projection.load_state_dict(state['pre_projection'])
 captured=[]
 handle=model.discriminator.register_forward_hook(lambda module,inputs,output:captured.append((-output).detach().cpu().numpy()))
 rows=[];start=time.perf_counter();parity=0
 with torch.inference_mode():
  for batch in loader:
   captured.clear();scores,*_=model._predict([x.to(device) for x in batch['image']]);b=len(scores)
   assert len(captured)==5
   patches=np.stack([x.reshape(b,-1) for x in captured],axis=1)
   flat=np.sort(patches.reshape(b,-1),axis=1)[:,::-1];perview=np.sort(patches,axis=2)[:,:,::-1]
   baseline=flat[:,0];parity=max(parity,float(np.max(np.abs(baseline-np.array(scores)))))
   values={'maximum':baseline}
   for f in FRACTIONS:
    values[f'global_top_{100*f:g}pct']=flat[:,:math.ceil(f*flat.shape[1])].mean(axis=1)
    values[f'view_top_{100*f:g}pct_max']=perview[:,:,:math.ceil(f*perview.shape[2])].mean(axis=2).max(axis=1)
   for i,name in enumerate(batch['image_path']):
    old=manifest[Path(name).name];label=int(batch['is_anomaly'][i]);assert label==int(old['label'])
    rows.append(dict(image=Path(name).name,label=label,split=old['split'],**{k:float(v[i]) for k,v in values.items()}))
   if len(rows)%200==0:print(f'{len(rows)}/4000, {time.perf_counter()-start:.1f}s',flush=True)
 handle.remove();assert len(rows)==4000 and parity<1e-6
 with (OUT/'scores.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 y=np.array([r['label'] for r in rows]);cal=np.array([r['split']=='calibration' for r in rows]);audit=~cal
 results={}
 for key in values:
  s=np.array([r[key] for r in rows]);t=recall_threshold(y[cal],s[cal],.9)
  results[key]=dict(threshold=t,calibration=metrics(y[cal],s[cal],t),audit=metrics(y[audit],s[audit],t))
 # Selection uses calibration only: minimize false alarms at >=90% recall, then higher AUROC, then stable name.
 selected=min(results,key=lambda k:(results[k]['calibration']['false_positive_rate'],-results[k]['calibration']['auroc'],k))
 old_scores=np.array([float(manifest[r['image']]['score']) for r in rows]);new_scores=np.array([r['maximum'] for r in rows])
 report=dict(protocol='Fixed retrospective calibration/audit split; no retraining. Includes maximum baseline and 12 predeclared alternatives. Selection: calibration FPR at >=90% recall, tie-break AUROC then name. Audit is development evidence, not fresh external validation.',device=str(device),seconds=time.perf_counter()-start,patches_per_view=int(patches.shape[2]),baseline_parity_max_abs=parity,cached_baseline_max_abs_difference=float(np.max(np.abs(old_scores-new_scores))),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),selected=selected,results=results)
 (OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 keys=['maximum',selected] if selected!='maximum' else ['maximum',min((k for k in results if k!='maximum'),key=lambda k:results[k]['calibration']['false_positive_rate'])]
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(3)
 for j,key in enumerate(keys):
  m=results[key]['audit'];axes[0].bar(x+(j-.5)*.36,[100*m[k] for k in ['auroc','recall','false_positive_rate']],.36,label=key)
  s=np.array([r[key] for r in rows])[audit]
  from sklearn.metrics import roc_curve
  fpr,tpr,_=roc_curve(y[audit],s);axes[1].plot(fpr,tpr,label=key)
 axes[0].set(xticks=x,xticklabels=['AUROC','Defect recall','Healthy false alarms'],ylabel='Percent',ylim=(0,105),title='Audit results: cutoffs chosen on calibration');axes[0].legend(fontsize=8)
 axes[1].set(xlabel='Healthy false-positive rate',ylabel='Defect recall',title='Audit ROC: unchanged weights');axes[1].legend(fontsize=8)
 fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
