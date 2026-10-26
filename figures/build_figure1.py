import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
out=Path(__file__).parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7,6)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,504);ax.set_ylim(432,0); ax.axis('off')
blue='#edf4f9'; orange='#fff7e9'; green='#edf6f1'; ink='#1b2e3a'; edge='#9eafba'
def text(x,y,s,size=8.5,weight='normal',color=ink,ha='left',va='center'):
 ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,linespacing=1.22)
def box(x,y,w,h,title,body='',fill=blue,title_size=9,body_size=8.5):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=4',lw=.7,edgecolor=edge,facecolor=fill))
 text(x+7,y+12,title,title_size,'bold')
 if body: text(x+7,y+24,body,body_size,va='top')
def arrow(points,color='#425868'):
 for i in range(len(points)-2):
  ax.plot([points[i][0],points[i+1][0]],[points[i][1],points[i+1][1]],color=color,lw=.85)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=7,lw=.85,color=color,shrinkA=0,shrinkB=1))
def head(y,s): text(9,y,s,10,'bold','#205c7c')
text(9,11,'Image-to-patient MES evaluation on LIMUC',12.5,'bold')
text(9,27,'ViT-B/16 and Swin-B use image-label supervision; evaluation paths remain distinct.',8.5)
head(46,'A  PATIENT-GROUPED NESTED DEVELOPMENT EVALUATION')
box(9,58,143,62,'Development cohort','9,590 images / 478 patients\n5 outer folds; hold out one\nKeep each patient together')
box(180,58,143,62,'4-fold inner selection','Select loss, aggregation,\nfamily and refit duration\nOnly outer-training patients')
box(351,58,144,62,'Refit and predict','Refit outer-training data\nPredict held-out outer fold\nNo outer-guided selection')
arrow([(152,89),(180,89)]);arrow([(323,89),(351,89)])
text(9,134,'Pool all five outer folds separately per seed; patient scoring uses modal-low (C).',8.5)
text(9,147,'Six aggregation candidates; all outer folds selected mean probability followed by argmax.',8.5)
head(169,'B  LOCKED OFFICIAL-PARTITION RE-EVALUATION')
box(9,181,128,54,'Official test images','1,686 images; 86 patients\nPreviously examined',fill=blue)
box(161,181,161,54,'15 checkpoints per family','5 development folds × 3 seeds\nHistorical fixed ensemble',fill=blue)
box(346,181,149,54,'Two views per checkpoint','Original + horizontal flip\nMean of 30 softmax vectors',fill=blue)
arrow([(137,208),(161,208)]);arrow([(322,208),(346,208)])
# Official averaged vector and two scoring branches.
box(9,256,128,66,'Image probabilities','Four-class image vector\nSeparate ViT and Swin\nprobability averages',fill=blue)
arrow([(421,235),(421,245),(73,245),(73,256)])
box(161,256,161,29,'Image argmax',fill=blue)
box(346,256,149,36,'Image-level scoring','Compare with image labels',fill=green)
arrow([(137,270),(161,270)]);arrow([(322,270),(346,270)])
box(161,296,161,40,'Patient mean → argmax','Average image probabilities',fill=blue)
box(346,296,149,40,'Patient-level scoring','Compare with modal-low (C)',fill=green)
arrow([(137,308),(149,308),(149,316),(161,316)]);arrow([(322,316),(346,316)])
text(9,348,'Official labels enter scoring only during re-evaluation; these are not nested outer refits.',8.5)
head(370,'C  REFERENCE CONSTRUCTION IS SEPARATE FROM PREDICTION')
box(9,382,143,37,'Released image labels','Expert MES labels 0–3',fill=orange)
box(180,382,143,37,'Modal-low reference','Modal grade; lower-grade ties',fill=orange,body_size=8.5)
box(351,382,144,37,'Patient-level comparison','Predictions from A or B',fill=green,body_size=8.5)
arrow([(152,401),(180,401)]);arrow([(323,401),(351,401)])
text(9,427,'Patient references summarize image bags, not independently adjudicated examination-level MES.',8.5)
for ext in ['svg','pdf','png']:
 fig.savefig(out/f'Figure1_EvaluationPaths_2026-10-21.{ext}',dpi=450,facecolor='white')
plt.close(fig)
