from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/GrainAD_Presentation_Improved.pdf'
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold')
W,H=960,540
NAVY='#142D3E'; TEAL='#007F86'; INK='#203846'; GREY='#526574'; PALE='#F5F7F7'; ORANGE='#AD4B22'
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('GrainAD: Five-view Paddy anomaly detection')
c.setAuthor('GrainAD project team')
page=0
def p(text,x,y,w,size=19,color=INK,bold=False):
    st=ParagraphStyle('p',fontName='ArialBold' if bold else 'Arial',fontSize=size,leading=size*1.3,textColor=HexColor(color))
    obj=Paragraph(text,st);_,h=obj.wrap(w,1000)
    assert y+h<510,(page,text,y,h)
    obj.drawOn(c,x,H-y-h)
    return h
def start(title,section='PROJECT',dark=False):
    global page
    page+=1
    c.setFillColor(HexColor(NAVY if dark else '#FFFFFF'));c.rect(0,0,W,H,fill=1,stroke=0)
    p(section,44,25,850,10,'#67CED0' if dark else TEAL,True)
    p(title,44,54,870,32,'#FFFFFF' if dark else NAVY,True)
def end(source='Local implementation and saved evaluation artifacts'):
    c.setFillColor(HexColor(GREY));c.setFont('Arial',8)
    c.drawString(44,18,source);c.drawRightString(916,18,f'{page:02d}')
    c.showPage()
def img(path,x,y,w,h):
    im=Image.open(path);iw,ih=im.size;s=min(w/iw,h/ih)
    c.drawImage(str(path),x+(w-iw*s)/2,H-y-(h+ih*s)/2,iw*s,ih*s,mask='auto')
def rows(items,y=145):
    for heading,body in items:
        p(heading,44,y,245,20,TEAL,True)
        h=p(body,312,y,600,19)
        y+=max(65,h+22)
def table(data,widths,y=155,size=18):
    def cell(v,head=False):
        return Paragraph(str(v),ParagraphStyle('t',fontName='ArialBold' if head else 'Arial',fontSize=size,leading=size*1.25,textColor=white if head else HexColor(INK)))
    t=Table([[cell(v,i==0) for v in row] for i,row in enumerate(data)],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor(PALE)]),('LINEBELOW',(0,-1),(-1,-1),.6,HexColor('#CCD8DA'))]))
    tw,th=t.wrap(872,1000);assert y+th<490,(page,th);t.drawOn(c,44,H-y-th)

start('GrainAD','PADDY ANOMALY DETECTION',True)
p('Inspecting one grain through five camera views',44,118,840,27,'#FFFFFF')
img(ROOT/'demo/samples/example_2.png',44,195,872,170)
p('A laptop implementation of the GrainBrain research method',44,402,870,20,'#FFFFFF')
p('Training, reproducible evaluation and visual prediction',44,440,870,17,'#A7C4D1')
end('Based on GrainNet/GrainBrain and Fan et al., IEEE TII, 2025')

start('The inspection problem','MOTIVATION')
rows([('One view is incomplete','A defect can appear on only one side of a grain. Five views provide more surface coverage.'),('Our task','Assign an anomaly score to each Paddy kernel and show regions that influence the prediction.'),('Training assumption','Learn from healthy grains and synthetic defects. Real defective samples support evaluation.'),('Intended use','Flag suspicious kernels for human review in this research demonstration.')])
end('Method: GrainBrain paper, Sections III-IV. Scope: Paddy software implementation only.')

start('What we built','CONTRIBUTION')
rows([('Research foundation','The authors supplied GrainAD, the backbone, spatial transformer and synthetic anomaly approach.'),('Our implementation','Corrected data handling and loss bookkeeping, trained the model, and added reliable restart support.'),('Our evaluation','Exported per-image scores, measured final-checkpoint performance and tested a calibrated cutoff.'),('Our deliverable','A portable prediction package, labeled heatmaps, source code and a detailed LaTeX report.')])
end('No matched training run of unmodified upstream code. Individual fixes have no measured causal AUROC gain.')

start('Data and training setup','EXPERIMENT')
table([['Item','Executed setting'],['Healthy training data','6,000 Paddy image strips, five views per strip'],['Evaluation data','4,000 test-image strips: 2,000 healthy + 2,000 defective'],['Synthetic defects','DTD textures and Perlin masks, plus Gaussian feature noise'],['Hardware','RTX 4050 Laptop GPU, 6 GB VRAM'],['Training budget','50 alignment epochs + 50 anomaly epochs, about 31 hours']], [270,602],y=137,size=18)
end('All supplied healthy training images used. Wall time includes an early restart. Pixel metrics remain unmeasured.')

start('How the model learns','TRAINING FLOW')
rows([('01  Prepare five views','Split each image strip. Resize each view to 224 x 224 and apply ImageNet normalization.'),('02  Learn alignment','Train a rotation spatial transformer using views of the same kernel.'),('03  Learn anomalies','Freeze alignment and WideResNet50. Train an adapter and classifier using real and synthetic features.'),('04  Save prediction weights','Keep alignment, adapter/classifier and pretrained backbone weights for inference.')],y=138)
end('Core files: data/grain.py, scripts/train_ica.py, scripts/train_src.py, models/simplenet.py')

start('Training objective and inference','MODEL DETAILS')
p('Loss = 0.5 x classification + 0.5 x contrastive learning',44,140,872,25,TEAL,True)
rows([('Classification loss','Distinguishes healthy patches from texture-based and noise-based synthetic anomalies.'),('Contrastive loss','Encourages compatible representations across views of the same kernel. Self-pairs are excluded.'),('Inference score','Take the maximum anomaly score across all patches and all five views of each kernel.')],y=210)
end('WideResNet50 layers 2-3. Gaussian feature noise standard deviation 0.015. Heatmap smoothing sigma 4.')

start('Architecture components','MODEL ARCHITECTURE')
rows([('Spatial transformer','Learns a rotation for each view. It aligns grain orientation before feature extraction.'),('WideResNet50 backbone','Uses ImageNet-pretrained layers 2 and 3 to extract visual features at different spatial scales.'),('Patch aggregation','Combines local 3 x 3 feature neighborhoods into a 1,536-dimensional representation per patch.'),('Adapter and classifier','A linear adapter specializes the features. A two-layer classifier predicts a normality logit for each patch.')],y=139)
end('Implementation: models/ICA.py, models/simplenet.py and configs/models/simplenet.yaml')

start('Which components actually learn?','TRAINABLE AND FROZEN WEIGHTS')
table([['Component','Alignment stage','Anomaly stage'],['Rotation STN','Train with alignment loss','Frozen'],['ImageNet backbone','No Paddy weight updates','Frozen'],['Linear feature adapter','Not trained in this stage','Train'],['Patch classifier','Not trained in this stage','Train']], [350,261,261],y=140,size=20)
p('Freezing large components reduces training memory and preserves pretrained features.',44,420,872,21,TEAL,True)
end('Adam updates STN first, then adapter and classifier. Inference performs no weight updates.')

start('How we create learning examples','SYNTHETIC ANOMALIES')
rows([('Healthy branch','Original healthy patches provide normal examples. The classifier learns high normality scores.'),('Image anomaly branch','Blend DTD textures into Perlin-mask regions. The mask supplies synthetic patch labels.'),('Feature anomaly branch','Add Gaussian noise to feature vectors, with standard deviation 0.015, to create anomalous examples.'),('At prediction time','Remove both synthesis branches. Run only the real image through alignment, feature extraction and scoring.')],y=139)
end('Synthetic labels enable training without real defective training examples. Synthetic defects may differ from real defects.')

start('Different kinds of learning in GrainAD','LEARNING OBJECTIVES')
rows([('Transfer learning','Reuse ImageNet features from a pretrained backbone instead of learning all visual features from scratch.'),('Self-supervised alignment','Use different views of the same kernel to learn rotation alignment without manual rotation labels.'),('Synthetic supervision','Binary classification learns healthy versus generated anomalous patches using known synthetic labels.'),('Contrastive learning','Group corresponding views by kernel identity to encourage consistent feature representations.')],y=139)
end('Stage 2 combines BCE and contrastive losses equally. Threshold calibration changes the decision rule, not neural weights.')

start('Changes that improve correctness','IMPLEMENTATION')
table([['Area','Change we made','Purpose'],['Data loader','Use all images with deterministic ordering','Avoid small random subsets'],['Contrastive loss','Correct view labels and exclude self-pairs','Match features to kernel identities'],['Alignment','Pair views from the same kernel','Avoid unrelated-grain comparisons'],['Inference','Aggregate views separately for every kernel','Remove whole-batch view selection'],['Masks and devices','Preserve dimensions and explicit device placement','Prevent shape and CPU/GPU errors']], [175,395,302],y=135,size=16)
end('Files: data/grain.py, models/loss.py, models/ICA.py, models/simplenet.py. Benefits are intended, not isolated ablation results.')

start('Changes that make training usable','ENGINEERING')
rows([('Recoverable training','Atomic checkpoints save epoch, optimizer and model state. Pause requests take effect at epoch boundaries.'),('Windows support','Worker initialization restores augmentation state and manages process-local random seeds.'),('Independent evaluation','A separate script measures the saved model and exports predictions, metrics and checkpoint identity.'),('Portable demonstration','CPU prediction, included weights and samples, plus regression coverage with 17 passing tests.')],y=139)
end('One early migration reset Adam after two alignment epochs. Worker-level randomness is not bit-exact across resume.')

start('Original maximum-score results','4,000-IMAGE TEST-IMAGE SET')
p('89.79%',44,139,400,66,TEAL,True)
p('Image AUROC',48,224,400,24,NAVY,True)
p('How well the model ranks defective grains above healthy grains across thresholds.',48,270,400,20)
table([['Metric','Result'],['Average precision','90.61%'],['Accuracy at cutoff 0','68.30%'],['Defect recall at cutoff 0','37.50%']], [265,175],y=139,size=18) if False else None
p('90.61% average precision',515,153,390,25,NAVY,True)
p('At the original zero cutoff',515,223,390,21,TEAL,True)
p('68.30% accuracy<br/>97.66% defect precision<br/>37.50% defect recall',515,267,390,23)
p('The default cutoff missed 1,250 of 2,000 defects.',44,425,872,23,ORANGE,True)
end('Source: results/paddy_full/evaluation/metrics.json. AUROC and accuracy measure different properties.')

start('Calibration changes the decision cutoff','NO RETRAINING')
rows([('Calibration half','2,000 images. Choose a cutoff that catches at least 90% of defects on this half.'),('Audit half','2,000 different images. Measure the chosen cutoff without using these images to select it.'),('Selected cutoff','Flag for review when the raw anomaly score is at least -2.0141.'),('Evidence limit','This split is retrospective. Development already inspected the parent test set, so it is not a fresh external test.')],y=139)
end('Stratified 50/50 split, seed 2026. Each half has 1,000 healthy and 1,000 defective kernels. Same weights throughout.')

start('More detected defects, more false alarms','SAME 2,000-IMAGE AUDIT HALF')
table([['Metric','Default cutoff','Calibrated cutoff'],['Defect recall','37.20%','89.90%'],['Defect precision','97.13%','72.91%'],['F1 score','53.80%','80.52%'],['Accuracy','68.05%','78.25%'],['Healthy false-alarm rate','1.10%','33.40%']], [390,241,241],y=130,size=18)
p('Audit AUROC stays 89.69%. Calibration changes decisions, not score ranking.',44,455,872,17,TEAL,True)
end('Source: results/paddy_full/calibration/threshold.json. Full-set AUROC of 89.79% uses a different sample population.')

start('The tradeoff in actual kernels','AUDIT COUNTS')
table([['Out of each group of 1,000','Default cutoff','Calibrated cutoff'],['Defective grains detected','372','899'],['Defective grains missed','628','101'],['Healthy grains falsely flagged','11','334'],['Healthy grains accepted','989','666']], [420,226,226],y=145,size=20)
p('527 additional defects detected, alongside 323 additional healthy grains flagged.',44,410,872,23,ORANGE,True)
end('Operational interpretation: review flags manually. The acceptable tradeoff depends on the cost of misses and false rejection.')

start('A detected defect','ORIGINAL MAXIMUM-SCORE DEMO')
img(ROOT/'outputs/demo/prediction_1.png',44,120,872,363)
end('Curated audit example. Heatmap intensity is relative within this image. Localization accuracy has not been measured.')

start('A missed defect','ORIGINAL MAXIMUM-SCORE DEMO')
img(ROOT/'outputs/demo/prediction_3.png',44,120,872,363)
end('The dataset labels this kernel defective, but the model accepts it. Examples explain behavior and do not estimate accuracy.')

start('Averaging suspicious patches improves ranking','NEW EXPERIMENT')
rows([('Original scoring','One maximum patch score determines the kernel decision. A single noisy patch can dominate.'),('New scoring','Average the top 5% of patch scores across all five views: 196 of 3,920 patches per kernel.'),('What changed','Only score aggregation and its calibrated cutoff. The neural-network weights stayed unchanged.'),('Why it may help','Averaging reduces the influence of isolated high scores. This explanation is a hypothesis supported by the observed result.')],y=139)
end('Experiment: scripts/test_patch_aggregation.py. Recomputed maximum scores exactly matched the saved baseline.')

start('How we selected the averaging rule','EXPERIMENT DESIGN')
rows([('Candidate rules','Tested 0.1%, 0.5%, 1%, 2%, 5% and 10% averages, globally and within each view: 12 alternatives.'),('Calibration only','For each rule, choose a cutoff for at least 90% recall. Select the rule with the fewest healthy false alarms.'),('Selected configuration','Global top-5% average, cutoff -3.50519. Calibration false alarms: 131 of 1,000 healthy grains.'),('Audit evaluation','Evaluate the selected configuration on the same separate 2,000-image audit half used for the baseline.')],y=139)
end('Retrospective development experiment. The parent dataset was inspected previously. Fresh external validation is still needed.')

start('Improved audit performance without retraining','SAME 2,000 IMAGES AND SAME WEIGHTS')
table([['Metric','Maximum score','Top-5% average'],['Image AUROC','89.69%','94.40%'],['Accuracy','78.25%','85.80%'],['Defect precision','72.91%','83.84%'],['F1 score','80.52%','86.20%'],['Defect recall','89.90%','88.70%'],['Healthy false-alarm rate','33.40%','17.10%']], [390,241,241],y=125,size=17)
p('AUROC improves by 4.71 percentage points because score ordering changes.',44,465,872,17,TEAL,True)
end('Both cutoffs selected on calibration data for 90% recall. Audit recall can differ from the calibration target.')

start('Visual evidence of improved separation','AUDIT COMPARISON')
img(ROOT/'results/paddy_full/aggregation_experiment/comparison.png',44,135,872,330)
end('Orange: global top-5% average. Blue: maximum score. Higher ROC curves indicate stronger ranking at a given false-alarm rate.')

start('Fewer false alarms, a small recall cost','COUNTS ON THE SAME AUDIT SET')
table([['Out of each group of 1,000','Maximum score','Top-5% average'],['Defects detected','899','887'],['Defects missed','101','113'],['Healthy grains falsely flagged','334','171'],['Healthy grains accepted','666','829']], [420,226,226],y=142,size=20)
p('163 fewer healthy false alarms, alongside 12 additional missed defects.',44,410,872,23,ORANGE,True)
end('The experimental rule has not yet replaced the live demo. Its threshold must travel with its scoring rule.')

start('Comparison with the original paper','PROTOCOL MATTERS')
table([['Paddy comparison','Paper GrainAD','Original rule','New rule'],['Image AUROC','94.5%','89.79%','94.40%'],['Evaluation population','test-pixel','test-image, 4,000','Audit half, 2,000'],['Pixel AUROC','84.6%','Not measured','Not measured']], [245,185,221,221],y=145,size=18)
p('Different evaluation populations prevent a claim that we matched or beat the paper.',44,405,872,22,ORANGE,True)
end('Paper: Table II and Section IV-A, DOI 10.1109/TII.2025.3528555. Protocol clarification: grainnet.github.io/GrainBrain.html')

start('What we would improve next','PRIORITIES')
rows([('First: fair comparison','Evaluate the current weights on test-pixel. Measure image and pixel performance under the authors\' protocol.'),('Then: reliable selection','Establish a separate validation set. Save checkpoints and choose the best validation result.'),('Then: controlled training','Test lower learning rates, contrastive loss choices and larger batches. Change one factor at a time.'),('Finally: new conditions','Test unseen acquisition sessions and realistic defect patterns before deployment.')],y=139)
end('These are proposed experiments. Smaller batches and training instability are hypotheses, not proven causes of lower scores.')

start('A working research demonstration','CONCLUSION',True)
p('Five views. One kernel-level decision.',44,145,872,34,'#FFFFFF',True)
p('We trained a complete Paddy anomaly detector on a laptop GPU and built a repeatable prediction workflow.',44,225,830,25,'#FFFFFF')
p('94.40% audit AUROC with top-5% averaging',44,330,870,28,'#67CED0',True)
p('Changing score aggregation improved audit ranking and reduced false alarms. Fresh data and a matched paper evaluation remain the next steps.',44,392,850,21,'#FFFFFF')
end('Discussion and demonstration')

start('Sources and reproducibility','REFERENCE SLIDE')
rows([('Original research','Fan et al., GrainBrain, IEEE Transactions on Industrial Informatics, 2025.<br/>DOI: 10.1109/TII.2025.3528555'),('Code and dataset','github.com/GrainNet/GrainBrain<br/>grainnet.github.io/GrainBrain.html'),('Measured results','results/paddy_full/evaluation/metrics.json<br/>results/paddy_full/calibration/threshold.json<br/>results/paddy_full/aggregation_experiment/results.json'),('Implementation record','report/main.tex and report/changes.patch<br/>Portable package: GrainAD_Paddy_Showcase.zip')],y=138)
end('Upstream commit c8ffa2039596e3212cf9becf3731064e5a7d3c81. Dataset: CC BY-NC-SA 4.0. Credit original authors.')
c.save()
print(f'{OUT}: {page} slides')
