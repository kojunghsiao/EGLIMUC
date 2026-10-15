from pathlib import Path
from html import escape
P=Path(__file__).parent
s=[]
s.append('''<svg xmlns="http://www.w3.org/2000/svg" width="7.2in" height="5.1in" viewBox="0 0 1600 1133"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 Z" fill="#35495b"/></marker></defs><rect width="1600" height="1133" fill="white"/>''')
def text(x,y,t,size=28,weight='normal',color='#152b3b',anchor='start'):
 s.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(t)}</text>')
def rect(x,y,w,h,fill='#f1f6fa',stroke='#a5b9c8',r=14):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def line(points,dashed=False):
 d='M'+' L'.join(f'{x},{y}' for x,y in points)
 s.append(f'<path d="{d}" fill="none" stroke="#35495b" stroke-width="3" marker-end="url(#arrow)"'+(' stroke-dasharray="9 7"' if dashed else '')+'/>')
def lines(x,y,ts,size=28,step=36,weight='normal',color='#152b3b'):
 for i,t in enumerate(ts):text(x,y+i*step,t,size,weight,color)
text(35,44,'Reliable evaluation of image-to-patient MES classification',40,'bold')
text(35,88,'How do prediction aggregation and the derived reference affect patient-level conclusions?',29,color='#4a5c69')
text(35,151,'A   IMAGE PREDICTION → PATIENT SUMMARY',29,'bold','#195c88')
# prediction lane
rect(35,175,285,262)
text(57,219,'LIMUC images',32,'bold')
# Image-card metaphor (not synthetic clinical image)
for x,y in [(67,241),(80,251),(93,261)]:
 rect(x,y,78,57,'#ffffff','#65879f',4)
 s.append(f'<path d="M{x+10},{y+43} L{x+29},{y+24} L{x+42},{y+36} L{x+56},{y+15} L{x+69},{y+43}" fill="none" stroke="#7c9bb0" stroke-width="3"/>')
lines(57,362,['Group all images','by patient identifier'],28,36)
line([(322,300),(366,300)])
rect(375,175,345,262)
text(397,219,'Image classifiers',32,'bold')
text(397,266,'ViT-B/16 or Swin-B',28)
text(397,307,'Class-weighted CE',28)
text(397,342,'+ optional ordinal term',27)
text(397,401,'Image-label supervision',27,color='#4a5c69')
line([(722,300),(766,300)])
rect(775,175,300,262)
text(797,219,'Image outputs',32,'bold')
lines(797,265,['MES probabilities','for each image (0–3)'],27,35)
for j,(lab,val) in enumerate([('0',.17),('1',.65),('2',.13),('3',.05)]):
 x=814+j*60
 s.append(f'<rect x="{x}" y="{386-val*105}" width="28" height="{val*105}" fill="#357ca4"/>')
 text(x+14,416,lab,25,anchor='middle')
line([(1077,300),(1121,300)])
rect(1130,175,435,262)
text(1152,219,'Patient prediction',32,'bold')
lines(1152,266,['Aggregate within each patient','Compare six candidate rules'],27,36)
text(1152,362,'Inner-selected rule:',27,'bold')
text(1152,401,'mean probability → argmax',28)
# reference lane
text(35,493,'B   DERIVED REFERENCE AND EVALUATION',29,'bold','#8b5c17')
rect(35,517,1040,241,'#fff8ed','#cbb78f')
text(57,562,'Released image labels',31,'bold')
lines(57,608,['MES 0–3 for the same','patient-image bag'],28,36)
line([(398,631),(443,631)])
text(470,562,'Derived patient reference',31,'bold')
text(470,608,'Primary: modal-low',29,'bold')
lines(470,653,['Sensitivity: modal-high,','floor / ceiling of median, maximum'],27,36)
# group branch from input to label box
line([(178,439),(178,465),(17,465),(17,631),(32,631)])
# evaluation connection
line([(1077,631),(1121,631)])
text(1098,605,'vs.',25,anchor='middle')
line([(1347,439),(1347,507)])
rect(1130,517,435,241,'#eff7f4','#94b3a4')
text(1152,562,'Patient-level evaluation',31,'bold')
lines(1152,608,['Ordinal agreement: QWK, MAE','Classes: macro-F1, accuracy','Probability quality: Brier, ECE'],26,39)
text(57,732,'Reference construction is separate from prediction aggregation.',26,color='#76531e')
# validation band
text(35,813,'C   PATIENT-GROUPED NESTED VALIDATION',29,'bold','#195c88')
rect(35,837,1530,212,'#f3f5f7','#aebbc5')
text(57,881,'5 outer folds',31,'bold')
lines(57,924,['Keep each patient’s images','together; hold out one fold'],28,36)
line([(456,923),(507,923)])
text(530,881,'4-fold inner selection',31,'bold')
lines(530,924,['Select loss, aggregation,','and model family'],28,36)
line([(935,923),(986,923)])
text(1010,881,'Refit → outer evaluation',31,'bold')
lines(1010,924,['Evaluate held-out patients;','pool predictions across outer folds'],28,36)
text(57,1023,'Post hoc diagnostics: cross aggregation × reference choices and inspect probability quality; no reselection.',26,color='#4a5c69')
text(35,1098,'Target: a derived patient-image-bag summary, not an independently adjudicated examination-level MES.',27,'bold')
s.append('</svg>')
(P/'Figure_1_conceptual_method_overview.svg').write_text('\n'.join(s))
