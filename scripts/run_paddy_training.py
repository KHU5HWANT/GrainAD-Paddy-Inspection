"""Run the full Paddy training stages sequentially and preserve logs/status."""
import json,os,subprocess,sys,time,msvcrt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'paddy_full'

def status(**fields):
    (OUT/'status.json').write_text(json.dumps(dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),**fields),indent=2))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pause_file=OUT/'pause.request'
    if '--pause' in sys.argv:
        pause_file.touch()
        print('Pause requested. Training will save and stop after its current epoch.')
        return
    lock=(OUT/'controller.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise SystemExit('A training controller is already active.')
    pause_file.unlink(missing_ok=True)
    env=os.environ.copy();env.update(GRAINAD_PAUSE_FILE=str(pause_file),TORCH_HOME=str(ROOT/'datasets'/'torch-cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TQDM_MININTERVAL='30',MPLCONFIGDIR=str(OUT/'matplotlib'))
    stages=[('alignment',['scripts/train_ica.py','classname=paddy','batch_size=2','num_workers=2','num_train_epochs=50','validation_epochs=1','save_dir=./results/paddy_full/ica']),
            ('grainad',['scripts/train_src.py','classname=paddy','batch_size=4','num_workers=2','models.meta_epochs=50','models.gan_epochs=1','ica_path=./results/paddy_full/ica/models/ica.pth','save_dir=./results/paddy_full/grainad','with_mask=false'])]
    for stage,args in stages:
        with (OUT/f'{stage}.log').open('a',encoding='utf-8') as log:
            log.write('\nSTART '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');log.flush()
            child=subprocess.Popen([sys.executable,'-B',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(state='running',stage=stage,pid=child.pid,controller_pid=os.getpid(),command=args)
            code=child.wait()
            if code == 75:
                status(state='paused',stage=stage,log=str(OUT/f'{stage}.log'))
                return
            if code:
                status(state='failed',stage=stage,exit_code=code,log=str(OUT/f'{stage}.log'))
                raise SystemExit(code)
    status(state='complete',stage='grainad',checkpoint=str(OUT/'grainad/models/grain/ckpt.pth'))

if __name__=='__main__':main()
