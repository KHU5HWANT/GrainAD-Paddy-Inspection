"""Create shareable, labeled prediction panels using the final Paddy model."""
import os,sys,json,argparse,hashlib,shutil,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME',str(ROOT/'datasets/torch-cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/matplotlib'))
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.simplenet import SimpleNet
from models.backbones import load

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',type=Path);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--output',type=Path,default=ROOT/'outputs/demo_top5');args=p.parse_args()
 torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
 settings=json.loads((ROOT/'demo/threshold.json').read_text());checkpoint=ROOT/'results/paddy_full/grainad/models/grain/ckpt.pth'
 with checkpoint.open('rb') as f:
  if hashlib.file_digest(f,'sha256').hexdigest()!=settings['checkpoint_sha256']:raise ValueError('This threshold belongs to a different checkpoint')
 if settings.get('aggregation')!='global_top_5pct' or settings.get('fraction')!=0.05:raise ValueError('Expected calibrated global top-5% scoring settings')
 threshold=settings['threshold'];device=torch.device(args.device)
 cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/models/simplenet.yaml'));cfg.update(device=device,input_shape=[3,224,224],ica_path=str(ROOT/'results/paddy_full/ica/models/ica.pth'),visualization=False)
 model=SimpleNet(device);model.load(backbone=load('wideresnet50'),**cfg)
 state=torch.load(checkpoint,map_location=device,weights_only=True);model.discriminator.load_state_dict(state['discriminator']);model.pre_projection.load_state_dict(state['pre_projection'])
 transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 examples=[dict(file=str(args.image.resolve()),label=None,case='User image')] if args.image else json.loads((ROOT/'demo/examples.json').read_text())
 summaries=[];panels=[]
 for i,example in enumerate(examples):
  path=Path(example['file']);path=path if path.is_absolute() else ROOT/path
  with Image.open(path) as image:
   image=image.convert('RGB');w,h=image.size
   if w<5:raise ValueError('Expected five horizontally concatenated views')
   step=w//5;views=[transform(image.crop((j*step,0,(j+1)*step if j<4 else w,h))).unsqueeze(0).to(device) for j in range(5)]
  captured=[]
  hook=model.discriminator.register_forward_hook(lambda module,inputs,output:captured.append((-output).detach().cpu().numpy()))
  try:
   _,maps,_,aligned,theta=model._predict(views)
  finally:
   hook.remove()
  if len(captured)!=5:raise ValueError('Expected patch scores from five views')
  patches=np.concatenate([x.reshape(-1) for x in captured]);k=math.ceil(settings['fraction']*len(patches))
  score=float(np.sort(patches)[::-1][:k].mean());heat=np.asarray(maps[0]);flagged=score>=threshold
  rgb=aligned[0].cpu().permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406]);rgb=np.clip(rgb,0,1)
  low,high=np.percentile(heat,[2,99]);relative=np.clip((heat-low)/max(high-low,1e-8),0,1)
  overlay=.65*rgb+.35*plt.get_cmap('inferno')(relative)[...,:3]
  truth='unknown' if example['label'] is None else ('defective' if example['label'] else 'healthy')
  case='User image' if example['label'] is None else ('Detected defect' if flagged else 'Missed defect') if example['label'] else ('False alarm' if flagged else 'Correctly accepted healthy')
  fig,axes=plt.subplots(2,1,figsize=(12,5),layout='constrained')
  fig.suptitle(f"{'FLAGGED FOR REVIEW' if flagged else 'ACCEPTED'}  |  score {score:.3f}  |  cutoff {threshold:.3f}",fontsize=17,fontweight='bold',color='#ad4b22' if flagged else '#007f86')
  axes[0].imshow(rgb);axes[0].set_title(f"Aligned five views | dataset label: {truth} | {case}",fontsize=11)
  axes[1].imshow(overlay);axes[1].set_title('Relative anomaly heatmap: brighter = more suspicious within this image; not a probability',fontsize=10)
  for ax in axes:ax.axis('off')
  fig.get_layout_engine().set(rect=(0,0.045,1,0.955))
  fig.text(.01,.012,'Top-5% patch average | retrospective calibration | localization accuracy not measured',fontsize=8,color='#555555')
  output=args.output/f'prediction_{i+1}.png';fig.savefig(output,dpi=160);plt.close(fig);panels.append(output)
  np.savez_compressed(args.output/f'prediction_{i+1}.npz',score=score,anomaly_map=heat,theta=torch.stack(theta).cpu().numpy())
  summaries.append(dict(image=path.name,score=score,threshold=threshold,decision='flagged' if flagged else 'accepted',ground_truth=truth,case=case,aggregation=settings['aggregation'],patch_count=len(patches),top_k=k,panel=output.name))
 widths=[];images=[]
 for path in panels:
  with Image.open(path) as im:images.append(im.convert('RGB').copy())
 canvas=Image.new('RGB',(max(im.width for im in images),sum(im.height for im in images)), 'white');y=0
 for im in images:canvas.paste(im,(0,y));y+=im.height
 canvas.save(args.output/'summary.png')
 (args.output/'predictions.json').write_text(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
