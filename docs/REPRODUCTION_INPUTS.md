# Controlled inputs for the archived saved-prediction audit

This code-only tree contains no archived input data. Authorized users must supply exact, unmodified files under `reproduction/inputs/`; public availability of the underlying images does not automatically authorize redistribution of these derived records.

Required top-level files:
- `nested_development_manifest.csv`: 9,590 development image rows; relative_path, patient_id, cv_fold, label, patient_y_true.
- `nested_metrics_by_seed.csv`: archived nine family/procedure-by-seed endpoint rows.
- `nested_bootstrap_samples_ViT_seed_42.csv`, `nested_bootstrap_samples_Swin_seed_42.csv`, and `nested_bootstrap_samples_NestedSelected_seed_42.csv`: archived bootstrap replicate tables.
- For each outer fold 0–4: `outer_fold_N_outer_fold_inner_selection_record.json`, `outer_fold_N_OUTER_FOLD_COMPLETED.json`, and `outer_fold_N_07_outer_refit_epoch_plan.csv`.
- For each fold N, family ViT/Swin, and seed 42/2025/2026: `outer_fold_N/FAMILY/seed_SEED/outer_blind_image_predictions.csv` and `RUN_COMPLETED.json`.

The blind prediction rows require relative_path, patient_id, y_pred, and prob_0 through prob_3; true labels are joined from the frozen manifest. Receipts provide exact prediction hashes, selection identifiers, epoch counts, and fingerprints. Do not change files or receipts to bypass a failed assertion. The scripts target the specific archived study run, not arbitrary alternative data.

`reproduce.py` writes `results/` and local verification/hash reports. Patient reference/prediction outputs remain sensitive derived records and are excluded by `.gitignore`. `make_summary_tables.py` consumes the generated CSVs. The official dataset alone is insufficient to reconstruct these saved model outputs without the corresponding training/inference artifacts.

No protected inputs are obtained or transmitted automatically. No permission to publish patient-level files is supplied by these instructions.
