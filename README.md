# EGLIMUC

Code accompanying the draft EGLIMUC study of ordinal ulcerative-colitis endoscopy classification and image-to-patient evaluation on LIMUC. This is an **unpublished, code-only release candidate**. Publication, repository version, author-approved citation, and code license are pending. No journal acceptance or external clinical validation is implied.

## What the study evaluates

Image-level ViT and Swin classifiers are trained with cross-entropy and ordinal-loss candidates. Image probabilities are aggregated into patient-image-bag summaries and compared with explicitly derived reference labels. A nested patient-grouped evaluation separates inner selection from outer evaluation. Crossed aggregation/reference analyses and probability diagnostics examine how the evaluation target changes conclusions.

A released patient image bag may contain images from multiple procedures or times. Its modal-low reference is a reproducible frequent-appearance summary, not an independently adjudicated contemporaneous patient diagnosis or a demonstrated clinically optimal target.

## Contents and notebook order

| Notebook | Role | Principal table mapping |
|---|---|---|
| 00 | Audit historical experiments and prior test exposure | S12 provenance |
| 01 | Freeze patient folds/reference definitions; ViT loss screening | I, II, IIIB, S1–S2, S4A |
| 02 | ViT multiseed confirmation | S4B–S4C |
| 03 | Patient aggregation and reference sensitivity | S2–S3 |
| 04 | Swin loss screening | II, S5A |
| 05 | Swin multiseed confirmation | S5B–S5C |
| 06 | Uniform-runtime backbone benchmark | IIIA–IIIB, S6 |
| 07 | Explanatory locked Swin ablation; no reselection | S7 |
| 08 | Reconstruct final model/checkpoint lock | IIIA, S12 |
| 09 | Evaluate the historically exposed official partition | V–VI, S8–S9, S11 |
| 10 | Earlier manuscript figures | Figure-generation provenance |
| 11 v2 | Corrected crossed aggregation/reference sensitivity | S10–S17 |
| 12 | Historical development-OOF probability audit | S18–S20 |
| 13 v2 corrected | Nested patient-grouped evaluation | IV, S21–S24; configuration in II/IIIB |

Read 00 first. The historical development chain is 01 → 02 → 03 → 04 → 05 → 06. Notebook 07 is an explanatory branch from 06; it does not choose the final model. The checkpoint-lock/evaluation chain is 06 → 08 → 09. Notebooks 10–12 consume the appropriate earlier saved artifacts. The latest conceptual Figure 1 is generated separately by `figures/build_figure.py`, not by the older Figure 1 code in Notebook 10.

Notebook 13 uses the frozen development manifests and performs its own inner selection within each outer-training partition. Do not substitute the historical globally selected candidate for its inner selection. Read its required-path and receipt checks before execution. Its default is `ASSEMBLE`, which requires complete saved fold outputs and **still writes files**. `RUN` explicitly enables training/resumption; `PLAN` is not a guarantee of a side-effect-free operation because setup/image/weight checks may run. The maximum documented workload is 360 inner fits plus 30 outer refits; runtime depends on hardware and is not benchmarked here.

The code in `reproduction/reproduce.py` produces the new nested posthoc diagnostics from archived blind predictions. `make_summary_tables.py` formats the diagnostic summaries. These scripts are necessary for S24–S28; Notebook 12 alone does not generate those new nested results.

Detailed, manually checkable mappings:
- [Notebook-to-table map](docs/notebook_table_map.csv)
- [Table-by-table source checklist](docs/table_source_checklist.csv)

Mappings include composite configuration/provenance tables; not every table is exported by one CSV. The checklist is for manual verification, not an assertion that every table was independently rerun. Table numbering follows the integrated October 2026 draft. The original double `.ipynb.ipynb` extension of Notebook 00 is normalized here; an older S12 alias should be reconciled when the manuscript is finalized.

## Reproducibility tiers and limitations

1. **Historical analysis:** Notebooks 00–12 document the original selection-conditioned analyses. The official partition was historically exposed. A later lock does not restore an untouched test set. Higher historical OOF scores are not interchangeable with nested performance.
2. **Nested analysis:** Corrected Notebook 13 implements five outer patient folds and four inner folds, with seeds 42, 2025, and 2026. The three seeds share seed-informed inner selection; they are not independent full selection-search replications. NestedSelected equals Swin in the archived run and is not a third independent model.
3. **Saved-output reproduction:** Before packaging, the archived 30 blind prediction files and corrected label joins were checked, and all 9 endpoint rows and 15,000 bootstrap replicate rows reproduced within 1e-14. An independent CPU run reproduced all 20 result CSVs byte-for-byte. This verifies saved-prediction evaluation, not original training, checkpoint weight bytes, fresh inference, or a clean full notebook run.
4. **Posthoc diagnostics:** The written plan was frozen before examining the new diagnostics but after primary results and review comments were known. It was not an original preregistration. Additional RPS/crossed metrics beyond that plan are exploratory. No new model, threshold, aggregation rule, or calibration fit was selected using these results. New diagnostic means/SDs do not imply new confidence intervals.

This public tree deliberately excludes patient-level inputs and therefore **is not self-contained for numerical reproduction**. See [required controlled inputs](docs/REPRODUCTION_INPUTS.md). Do not upload the owner/reviewer reproduction ZIP unchanged.

## Data and workspace setup

Obtain LIMUC from the official [Zenodo record](https://doi.org/10.5281/zenodo.5827695), following its applicable terms. This repository does not redistribute images, labels, patient manifests, image filenames, saved patient/image predictions, or model weights. The audited current release contains 9,590 development images from 478 filename-derived patient IDs and 1,686 official-partition images from 86 IDs. The difference from the original publication's 479/85 split is documented, not assumed to be a confirmed publication error. Filename IDs are not independent biological identity verification.

The notebooks retain their Colab-oriented setup and dependency-install commands. Inspect them before running; some import `google.colab`. On Colab, set `EGLIMUC_WORKSPACE` in the first cell to your own workspace, for example `/content/eglimuc`. Expected layout under that workspace:

    LIMUC/train_and_validation_sets/
    LIMUC/test_set/                       # historical Notebook 09 only
    LIMUC/ViT/                           # optional legacy artifacts for 00
    LIMUC/ViT2/revision_2026/             # frozen artifacts and stage outputs

The underlying archive layout and saved manifest paths must match the notebook's actual checks. Notebook 00 may require additional legacy checkpoint/output subdirectories. The first cell uses `setdefault`; it does not override a value you already configured. Personal Drive-root literals have been replaced by this environment variable. Optional Drive mounting is disabled unless `EGLIMUC_MOUNT_DRIVE=1`; if you enable it, set `EGLIMUC_WORKSPACE` to your actual mounted root yourself. No private owner's Drive access links are needed.

Use a fresh output workspace and retain immutable archival evidence. Notebooks can write or replace outputs, resume checkpoints, download pretrained weights, install dependencies, or clean temporary/completed checkpoint files. Path changes can alter path-bearing manifests or fingerprints: never edit archived receipts or fingerprints merely to force a mismatch to pass. Retain original evidence for saved-output verification and create a new provenance record for a fresh training run.

## Dependencies and checks

`reproduction/requirements.txt` records the verified CPU evaluation environment (Python 3.12; NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0). It is **not** a complete historical training lockfile. Training notebooks retain their evidenced installation specifications, including timm 1.0.28 in relevant stages and historical pandas/scikit-learn/Pillow pins. PyTorch/CUDA and some historical patch versions are not independently established; do not present a newly installed environment as the exact original environment.

Run the small synthetic checks with `python tests/test_public_helpers.py`. They do not establish full notebook execution or training reproducibility.

Run CPU reproduction only after obtaining authorized exact inputs:

    python reproduction/reproduce.py
    python reproduction/make_summary_tables.py

These scripts write outputs, including patient-level files; keep `reproduction/results/` private. They do not train or access the network. `python figures/build_figure.py` writes the conceptual SVG; the included PDF/PNG are rendered assets. The generator itself produces only SVG and uses Python standard-library modules. Release cleanup was syntax-checked and reviewed without running training or installing packages. Source hashes and cleaned-file hashes are recorded in `docs/SOURCE_PROVENANCE.json`; hashes identify different original and sanitized bytes, not identical files.

## Release hygiene, citation, and rights

Saved cell outputs, execution counts, widget state, personal Colab metadata, and private source links have been removed from release copies. Original archived sources were retained separately. Algorithmic functions and scientific configuration were preserved; changes concern workspace configuration and optional Drive mounting. The corrected Notebook 13 contains the duplicate-column label-merge/recovery fixes rather than the older appended recovery notebook.

Code license selection is pending author approval. No MIT, Apache, or other open-source rights grant is asserted by this repository. Review third-party code, pretrained-model, and dataset terms separately. Do not infer redistribution permission from citation alone. Authors must approve the final repository and citation before publication. Until then cite descriptively as “EGLIMUC, unpublished research code, release candidate”; no journal DOI, final author list, or publication year is invented here.

## Deliberately excluded investigations

Notebooks 14, 14b, and 14c are optional data-integrity investigations, not core numbered-table generators. Their archived scripts depend on private inventory/configuration files and are not bundled here. The official archive directory audit behind Supplementary Methods S2.9 is consequently documented in the manuscript but not independently rerunnable from this code-only tree. Notebook 15 and the patient-balanced training notebook 16 are exploratory work outside the current paper and are also excluded.
