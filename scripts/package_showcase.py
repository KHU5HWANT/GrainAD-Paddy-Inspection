from pathlib import Path
import hashlib,json,zipfile,subprocess,shutil
root=Path(__file__).resolve().parents[1]
files=set()
for folder in ('models','utils','configs','data','scripts','tests','demo','report','Presentation','outputs/demo','outputs/demo_top5','results/paddy_full/evaluation','results/paddy_full/calibration'):
 for p in (root/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.log'):
   files.add(p)
for name in ('ATTRIBUTION.txt','requirements.txt','requirements-demo.txt','RUN_DEMO.cmd','SETUP_WINDOWS.cmd','PROJECT_GUIDE.txt','results/paddy_full/ica/models/ica.pth','results/paddy_full/grainad/models/grain/ckpt.pth','datasets/torch-cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth'):
 files.add(root/name)
for source,target in [('prediction_1.png','demo_detected.png'),('prediction_3.png','demo_missed.png')]:
 shutil.copy2(root/'outputs/demo'/source,root/'report/images'/target)
(root/'report/changes.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--','.'],cwd=root))
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
out=root.parent/'Share'/'GrainAD_Paddy_Showcase.zip';out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for p in sorted(files):z.write(p,'GrainAD_Paddy/'+p.relative_to(root).as_posix())
 z.writestr('GrainAD_Paddy/SHA256_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
print(json.dumps({'zip':str(out),'files':len(files),'size_mb':round(out.stat().st_size/1e6,2),'integrity':'passed'}))
