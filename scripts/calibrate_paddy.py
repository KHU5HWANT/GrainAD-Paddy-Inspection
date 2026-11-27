"""Reproducible retrospective calibration; audit data do not select the cutoff."""
import sys,json,csv,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve
from utils.calibration import recall_threshold

ROOT=Path(__file__).resolve().parents[1]
def metrics(y,s,t):
    p=s>=t;cm=confusion_matrix(y,p,labels=[0,1])
    return dict(auroc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),accuracy=float(accuracy_score(y,p)),precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p)),f1=float(f1_score(y,p)),false_positive_rate=float(cm[0,1]/cm[0].sum()),confusion_matrix=cm.tolist())
def main():
    src=ROOT/'results/paddy_full/evaluation';out=ROOT/'results/paddy_full/calibration';out.mkdir(exist_ok=True)
    rows=sorted(list(csv.DictReader((src/'predictions.csv').open())),key=lambda r:r['image'])
    assert len({r['image'] for r in rows})==len(rows)
    y=np.array([int(r['is_defective']) for r in rows]);s=np.array([float(r['anomaly_score']) for r in rows])
    cal,audit=train_test_split(np.arange(len(rows)),test_size=.5,random_state=2026,stratify=y)
    threshold=recall_threshold(y[cal],s[cal],.9)
    result=dict(protocol='Retrospective stratified 50/50 split of the previously inspected test-image set; not a pristine external test.',seed=2026,target_calibration_recall=.9,calibration_n=len(cal),audit_n=len(audit),threshold=threshold,checkpoint_sha256=json.loads((src/'metrics.json').read_text())['checkpoint_sha256'],calibration=metrics(y[cal],s[cal],threshold),audit_default=metrics(y[audit],s[audit],0),audit_calibrated=metrics(y[audit],s[audit],threshold))
    (out/'threshold.json').write_text(json.dumps(result,indent=2))
    audit_set=set(audit.tolist())
    with (out/'split_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image','label','score','split'])
        for i,r in enumerate(rows):w.writerow([r['image'],int(y[i]),float(s[i]),'audit' if i in audit_set else 'calibration'])
    # Show successes and failures deliberately, with the selection rule disclosed.
    examples=[]
    for label,pred,title in [(1,1,'Detected defect'),(0,0,'Correctly accepted healthy'),(1,0,'Missed defect'),(0,1,'False alarm')]:
        candidates=[i for i in sorted(audit) if y[i]==label and int(s[i]>=threshold)==pred]
        if candidates:
            i=candidates[0];examples.append(dict(source=rows[i]['image'],label=label,case=title,cached_score=float(s[i])))
    (out/'demo_examples.json').write_text(json.dumps(examples,indent=2))
    images=ROOT/'report/images';images.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    fpr,tpr,_=roc_curve(y[audit],s[audit]);axes[0].plot(fpr,tpr,color='#007f86',lw=2);axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False-positive rate',ylabel='Defect recall',title=f"Audit ROC | AUC {result['audit_calibrated']['auroc']:.3f}")
    for v,name,color in [(0,'Healthy','#007f86'),(1,'Defective','#d26939')]:axes[1].hist(s[audit][y[audit]==v],bins=35,alpha=.55,label=name,color=color)
    axes[1].axvline(threshold,color='black',ls='--',label='Calibrated cutoff');axes[1].set(title='Audit score distributions',xlabel='Raw anomaly score',ylabel='Images');axes[1].legend(fontsize=8)
    keys=['recall','precision','accuracy'];x=np.arange(3)
    axes[2].bar(x-.18,[result['audit_default'][k]*100 for k in keys],.36,label='Default cutoff',color='#abb9c4')
    axes[2].bar(x+.18,[result['audit_calibrated'][k]*100 for k in keys],.36,label='Calibrated cutoff',color='#007f86');axes[2].set(xticks=x,xticklabels=['Recall','Precision','Accuracy'],ylim=(0,105),ylabel='Percent',title='Same audit set; unchanged weights');axes[2].legend(fontsize=8)
    fig.savefig(images/'performance.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,key,title in zip(axes,['audit_default','audit_calibrated'],['Default cutoff','Calibrated cutoff']):
        cm=np.array(result[key]['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=1000)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>500 else 'black',fontsize=16)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Healthy','Defective'],yticklabels=['Healthy','Defective'],xlabel='Predicted',ylabel='Actual',title=title)
    fig.savefig(images/'confusion.png',dpi=180);plt.close(fig)
    log=(ROOT/'results/paddy_full/grainad.log').read_text(errors='replace');points=[(int(e)+1,float(v)) for e,v in re.findall(r'----- (\d+) I-AUROC:([0-9.]+)',log)]
    fig,ax=plt.subplots(figsize=(9,2.8),layout='constrained');ax.plot(*zip(*points),color='#007f86');ax.set(xlabel='Completed epoch',ylabel='Image AUROC',ylim=(0,1),title='Training history: substantial epoch-to-epoch variability');fig.savefig(images/'history.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
