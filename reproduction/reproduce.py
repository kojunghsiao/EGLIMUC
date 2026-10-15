"""CPU-only frozen-prediction evaluation; run: python reproduce.py. No training or network."""
from pathlib import Path
import ast,json,hashlib,sys,platform,re,math
import numpy as np,pandas as pd
from sklearn.metrics import cohen_kappa_score,confusion_matrix,mean_absolute_error,f1_score,accuracy_score
ROOT=Path(__file__).resolve().parent; IN=ROOT/'inputs'; OUT=ROOT/'results'
if not (IN/'nested_development_manifest.csv').is_file():
 raise SystemExit('Required controlled inputs are absent. See docs/REPRODUCTION_INPUTS.md. This is a code-only release.')
OUT.mkdir(exist_ok=True)
NUM_CLASSES=4;CLASS_IDS=list(range(4));PROBABILITY_COLUMNS=[f'prob_{i}' for i in CLASS_IDS];PROBABILITY_EPSILON=1e-12

def import_functions(path,names):
 tree=ast.parse(path.read_text()); nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
 assert {x.name for x in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
import_functions(ROOT/'notebook13_corrected_source.py',['safe_qwk','fixed_balanced_accuracy','compute_metrics','normalize_probability_vector','round_half_up','round_half_up_to_class','aggregate_one_patient','aggregate_image_predictions','stratified_bootstrap','normalize_path','canonical_integer_column','attach_labels'])
import_functions(ROOT/'12_development_oof_probability_calibration_audit.py',['equal_width_bin_ids','weighted_mean','build_calibration_precomputation','binned_calibration_statistics','calibration_metrics_from_precomputation'])
outer_label_vault_df=pd.read_csv(IN/'nested_development_manifest.csv');m=outer_label_vault_df
assert len(m)==9590 and m.patient_id.nunique()==478 and not m.relative_path.duplicated().any()
assert m.groupby('patient_id').cv_fold.nunique().max()==1
for fold in range(5):
 rec=json.load(open(IN/f'outer_fold_{fold}_outer_fold_inner_selection_record.json'))
 complete=json.load(open(IN/f'outer_fold_{fold}_OUTER_FOLD_COMPLETED.json'))
 assert rec['analysis_plan_fingerprint']==complete['analysis_plan_fingerprint']=='5c5cab052235f11e10a43f0fc23cdc8831a89b502bcbefa85973ad6e5d076cd6'
 assert rec['selected_system']=='Swin' and rec['selected_aggregation_rule']=='mean_probability_argmax'
 assert complete['n_completed_outer_refits']==6
 plan=pd.read_csv(IN/f'outer_fold_{fold}_07_outer_refit_epoch_plan.csv')
 for row in plan.itertuples():
  receipt=json.load(open(IN/f'outer_fold_{fold}'/row.family/f'seed_{row.nominal_seed}'/'RUN_COMPLETED.json'))
  assert receipt['candidate_id']==row.candidate_id==rec['selected_candidate_by_family'][row.family]
  assert receipt['refit_epochs']==row.outer_refit_epochs==rec['refit_epochs_by_family_and_seed'][row.family][str(row.nominal_seed)]
refs=[]
for pid,g in m.groupby('patient_id',sort=True):
 y=g.label.to_numpy();counts=np.bincount(y,minlength=4);ties=np.flatnonzero(counts==counts.max());med=np.median(y)
 refs.append(dict(patient_id=pid,modal_low=ties[0],modal_high=ties[-1],median_low=int(np.floor(med)),median_high=int(np.ceil(med)),maximum_label=int(y.max())))
refs=pd.DataFrame(refs); assert np.array_equal(m.groupby('patient_id').patient_y_true.first().values,refs.modal_low.values)
refs.to_csv(OUT/'patient_references.csv',index=False)
refs.drop(columns='patient_id').apply(lambda x:x.value_counts().reindex(CLASS_IDS,fill_value=0)).to_csv(OUT/'reference_counts.csv',index_label='MES')
metrics=[];confusions=[];cross=[];cal=[];reliability=[];selections=[];checks=[];patients={}
rules=['mean_probability_argmax','median_probability_argmax','majority_vote_low_tie','mean_expected_mes_round','median_expected_mes_round','maximum_predicted_image_class']
for fam in ['ViT','Swin']:
 for seed in [42,2025,2026]:
  frames=[]
  for fold in range(5):
   d=IN/f'outer_fold_{fold}'/fam/f'seed_{seed}';f=d/'outer_blind_image_predictions.csv';raw=pd.read_csv(f);receipt=json.load(open(d/'RUN_COMPLETED.json'))
   sha=hashlib.sha256(f.read_bytes()).hexdigest();assert sha==receipt['blind_prediction_sha256'],f'SHA mismatch {f}'
   assert receipt['analysis_plan_fingerprint']=='5c5cab052235f11e10a43f0fc23cdc8831a89b502bcbefa85973ad6e5d076cd6'
   assert set(raw.relative_path)==set(m.loc[m.cv_fold==fold,'relative_path']); assert not raw.relative_path.duplicated().any()
   assert set(raw.patient_id)==set(m.loc[m.cv_fold==fold,'patient_id'])
   assert np.allclose(raw[PROBABILITY_COLUMNS].sum(axis=1),1,atol=1e-5)
   before=raw[PROBABILITY_COLUMNS].to_numpy().copy();labeled=attach_labels(raw)
   assert np.array_equal(before,labeled[PROBABILITY_COLUMNS].to_numpy())
   frames.append(labeled);checks.append(dict(family=fam,seed=seed,fold=fold,n_images=len(raw),n_patients=raw.patient_id.nunique(),sha256=sha,join='PASS',probabilities_unchanged=True))
   selections.append({k:receipt[k] for k in ['outer_fold','family','candidate_id','nominal_seed','refit_epochs','run_fingerprint','analysis_plan_fingerprint']})
  allimages=pd.concat(frames,ignore_index=True);assert len(allimages)==9590 and not allimages.relative_path.duplicated().any()
  p=aggregate_image_predictions(allimages,'mean_probability_argmax');patients[(fam,seed)]=p;p.to_csv(OUT/f'patient_predictions_{fam}_{seed}.csv',index=False)
  met={'nested_method':fam,'nominal_seed':seed,**{'patient_'+k:v for k,v in compute_metrics(p.y_true,p.y_pred).items()}}
  im=compute_metrics(allimages.y_true,allimages.y_pred);met.update(image_qwk=im['qwk'],image_mae=im['mae']);metrics.append(met)
  cm=confusion_matrix(p.y_true,p.y_pred,labels=CLASS_IDS)
  for i in CLASS_IDS:
   for j in CLASS_IDS:confusions.append(dict(family=fam,seed=seed,true=i,predicted=j,count=int(cm[i,j])))
  for rule in rules:
   agg=aggregate_image_predictions(allimages,rule).merge(refs,on='patient_id',validate='one_to_one')
   for ref in refs.columns[1:]:cross.append(dict(family=fam,seed=seed,aggregation=rule,reference=ref,**compute_metrics(agg[ref],agg.y_pred)))
  for unit,reference,yt,prob in [('image','image_label',allimages.y_true,allimages[PROBABILITY_COLUMNS])]+[('patient',r,refs[r],p[PROBABILITY_COLUMNS]) for r in refs.columns[1:]]:
   pre=build_calibration_precomputation(yt,prob,15);vals=calibration_metrics_from_precomputation(pre);cal.append(dict(family=fam,seed=seed,unit=unit,reference=reference,n=len(yt),**vals))
   for b in range(15):
    sel=pre['top_bin_ids']==b;n=sel.sum();reliability.append(dict(family=fam,seed=seed,unit=unit,reference=reference,bin=b,lower=b/15,upper=(b+1)/15,n=int(n),confidence=float(pre['confidence'][sel].mean()) if n else np.nan,accuracy=float(pre['correct'][sel].mean()) if n else np.nan))
  print('Evaluated',fam,seed,flush=True)
metrics +=[{**r,'nested_method':'NestedSelected'} for r in metrics if r['nested_method']=='Swin']
out=pd.DataFrame(metrics); archived=pd.read_csv(IN/'nested_metrics_by_seed.csv');a=out.sort_values(['nested_method','nominal_seed']).reset_index(drop=True);b=archived.sort_values(['nested_method','nominal_seed']).reset_index(drop=True)
assert np.allclose(a[b.columns[2:]],b[b.columns[2:]],atol=1e-14,rtol=0)
out.to_csv(OUT/'reproduced_metrics_by_seed.csv',index=False)
for name,data in [('confusion_counts',confusions),('crossed_diagnostics',cross),('calibration_diagnostics',cal),('reliability_bins',reliability),('outer_refit_selection_epochs',selections),('prediction_integrity_checks',checks)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
boot=[]
for idx,method in enumerate(['ViT','Swin','NestedSelected']):
 p=patients[('Swin' if method=='NestedSelected' else method,42)];samples=stratified_bootstrap(p,5000,20260905+idx*10000);old=pd.read_csv(IN/f'nested_bootstrap_samples_{method}_seed_42.csv');assert np.allclose(samples[old.columns],old,rtol=0,atol=1e-14)
 for metric in ['qwk','mae','macro_f1']:
  lo,hi=np.percentile(samples[metric],[2.5,97.5]);boot.append(dict(method=method,metric=metric,lower=lo,upper=hi,seed=20260905+idx*10000))
 print('Bootstrap exact match',method,flush=True)
pd.DataFrame(boot).to_csv(OUT/'reproduced_bootstrap_intervals.csv',index=False)
summary=out.groupby('nested_method').agg({c:['mean','std'] for c in out.columns if c not in ['nested_method','nominal_seed']});summary.to_csv(OUT/'reproduced_multiseed_summary.csv')
hashes=[dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in sorted(IN.rglob('*')) if f.is_file()];pd.DataFrame(hashes).to_csv(ROOT/'input_sha256_manifest.csv',index=False)
(ROOT/'VERIFICATION.txt').write_text('PASS: 30 prediction SHA256 hashes match archived completion receipts; 30 one-to-one label joins; 9590 images/478 patients per family/seed; patient folds disjoint; numeric probability values unchanged by corrected attach_labels; all 9 archived per-seed endpoint rows reproduced within 1e-14; all 15000 archived bootstrap replicate rows reproduced within 1e-14. No training or inference rerun. Checkpoint weights and inner-training provenance not independently re-executed.\nPython '+sys.version+'\nNumPy '+np.__version__+'; pandas '+pd.__version__+'\n')
print('ALL CHECKS PASS',flush=True)
