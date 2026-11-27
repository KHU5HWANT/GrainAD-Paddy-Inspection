"""Evaluate final Paddy checkpoint without fitting a threshold on test data."""
import sys,os,time,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME',str(ROOT/'datasets/torch-cache'))
import torch,numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from omegaconf import OmegaConf
from data.grain import GrainDataset,initialize_worker
from models.simplenet import SimpleNet
from models.backbones import load

def main():
 torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 out=ROOT/'results/paddy_full/evaluation';out.mkdir(parents=True,exist_ok=True)
 ds=GrainDataset(str(ROOT/'datasets/grain'),classname='paddy',is_train=False,five_split=True,with_mask=False,resize=[224,224],imagesize=[224,224],padding=0,anomaly_source_path='',generate_anomalies=False)
 loader=DataLoader(ds,batch_size=4,num_workers=2,worker_init_fn=initialize_worker)
 cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/models/simplenet.yaml'))
 cfg.update(device=device,input_shape=[3,224,224],ica_path=str(ROOT/'results/paddy_full/ica/models/ica.pth'),visualization=False)
 model=SimpleNet(device);model.load(backbone=load('wideresnet50'),**cfg)
 path=ROOT/'results/paddy_full/grainad/models/grain/ckpt.pth';state=torch.load(path,map_location=device,weights_only=True)
 model.discriminator.load_state_dict(state['discriminator']);model.pre_projection.load_state_dict(state['pre_projection'])
 rows=[];start=time.perf_counter()
 with torch.inference_mode():
  for batch in loader:
   views=[x.to(device) for x in batch['image']]
   scores,*_=model._predict(views)
   for name,label,score in zip(batch['image_path'],batch['is_anomaly'].tolist(),scores):rows.append((name,label,float(score)))
   if len(rows)%200==0:print(f'Evaluated {len(rows)}/{len(ds)} in {time.perf_counter()-start:.1f}s',flush=True)
 elapsed=time.perf_counter()-start;y=np.array([x[1] for x in rows]);s=np.array([x[2] for x in rows]);pred=s>=0
 result=dict(dataset='Paddy test-image',samples=len(y),normal=int((y==0).sum()),defective=int((y==1).sum()),image_auroc=roc_auc_score(y,s),average_precision=average_precision_score(y,s),threshold=0,threshold_note='Uncalibrated default negative-logit cutoff; not selected on test data.',accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),seconds=elapsed,kernels_per_second=len(y)/elapsed,device=str(device),pixel_metrics='not evaluated')
 with path.open('rb') as f:result['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
 (out/'metrics.json').write_text(json.dumps(result,indent=2))
 with (out/'predictions.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['image','is_defective','anomaly_score']);w.writerows(rows)
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
