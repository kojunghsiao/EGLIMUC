from pathlib import Path
import pandas as pd
r=Path(__file__).resolve().parent/'results'
e=pd.read_csv(r/'outer_refit_selection_epochs.csv')
w=e.pivot(index=['outer_fold','family','candidate_id'],columns='nominal_seed',values='refit_epochs').reset_index()
w.columns=[str(c) for c in w.columns];w=w.rename(columns={'42':'epochs_seed42','2025':'epochs_seed2025','2026':'epochs_seed2026'})
w['aggregation']='mean_probability_argmax';w['selected_family']='Swin';w.to_csv(r/'selection_table_compact.csv',index=False)
for name,keys in [('crossed_diagnostics',['family','aggregation','reference']),('calibration_diagnostics',['family','unit','reference'])]:
 d=pd.read_csv(r/(name+'.csv'));cols=[c for c in d.select_dtypes('number').columns if c not in ['seed','n']]
 s=d.groupby(keys)[cols].agg(['mean','std']);s.columns=['_'.join(c) for c in s.columns];s.to_csv(r/(name+'_mean_sd.csv'))
