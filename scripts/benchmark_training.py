"""Time real Paddy batches; benchmark updates are NOT production checkpoints."""
import sys, time, json, argparse, gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
from data.grain import GrainDataset, initialize_worker
from models.ICA import STNModule

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['ica','ad'],required=True);p.add_argument('--batch',type=int,default=2);p.add_argument('--steps',type=int,default=8);p.add_argument('--workers',type=int,default=0);a=p.parse_args()
    assert torch.cuda.is_available(), 'CUDA is required for this benchmark'
    torch.set_num_threads(4)
    ds=GrainDataset('datasets/grain',classname='paddy',five_split=True,resize=[224,224],imagesize=[224,224],padding=0,anomaly_source_path='datasets/dtd/images',generate_anomalies=a.stage=='ad')
    loader=DataLoader(Subset(ds,list(range(a.batch*a.steps))),batch_size=a.batch,num_workers=a.workers,persistent_workers=a.workers>0,worker_init_fn=initialize_worker)
    device=torch.device('cuda'); stn=STNModule(3).to(device)
    if a.stage=='ica':
        opt=torch.optim.Adam(stn.parameters(),lr=1e-4)
        times=[]; mark=time.perf_counter()
        for batch in loader:
            x=torch.cat(batch['image']).to(device); views=torch.stack(batch['image']).to(device)
            opt.zero_grad(set_to_none=True)
            y,_=stn(x); z,_=stn(views.roll(1,0).flatten(0,1))
            loss=torch.nn.functional.mse_loss(y,z);loss.backward();opt.step();torch.cuda.synchronize()
            times.append(time.perf_counter()-mark);mark=time.perf_counter()
        torch.save(stn.state_dict(),'results/benchmark/ica.pth')
        duration=sum(times[2:])/len(times[2:])
    else:
        del stn;gc.collect();torch.cuda.empty_cache()
        from models.simplenet import SimpleNet
        from models.backbones import load
        from omegaconf import OmegaConf
        cfg=OmegaConf.to_container(OmegaConf.load('configs/models/simplenet.yaml'))
        cfg.update(device=device,input_shape=[3,224,224],ica_path='results/benchmark/ica.pth',gan_epochs=1,visualization=False)
        model=SimpleNet(device); model.load(backbone=load('wideresnet50'),**cfg)
        # Warm-up then measure the actual loader, synthesis, forward/backward, and optimizer.
        warm_iter=iter(loader); warm=[next(warm_iter),next(warm_iter)]
        model._train_discriminator(warm);torch.cuda.synchronize()
        mark=time.perf_counter();model._train_discriminator(loader);torch.cuda.synchronize()
        duration=(time.perf_counter()-mark)/len(loader)
    result=dict(stage=a.stage,batch_size=a.batch,workers=a.workers,seconds_per_batch=duration,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,paddy_epoch_minutes=duration*((len(ds)+a.batch-1)//a.batch)/60)
    Path(f'results/benchmark/{a.stage}_b{a.batch}_w{a.workers}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
