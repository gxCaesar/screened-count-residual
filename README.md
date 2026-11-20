# Screened Count Residual: Spatial Threshold Calibration for Virtual Immunostaining

Standalone, CPU-verifiable reference implementation, worked calibration example, split manifests, and confirmatory per-unit result tables for gauge-fixed Screened Count Residuals across 20 pretrained virtual-staining generator instances spanning three cohorts (`HEMIT`, `Harvard Orion CRC`, and `HER2match`).

## Repository Layout

- `s2/`: Complete transitive import closure of core operators (`query_relation_reconstruction.py`, `query_relation_count_objective.py`), plug-and-play universal residual head (`screened_universal_head.py`), generator adapters (`backbone_zoo.py`), single-cell spatial proximity and duplicate-free survival evaluator (`clinical_cell_survival.py`), confirmatory campaign runners (`run_hemit_multibackbone_cell.py`, `run_orion_multibackbone_survival.py`, `run_her2match_dab_campaign.py`), and *Nature Methods* expansion runners (`run_nm_comparators20.py`, `run_orion_5fold_16ch.py`, `run_nm_biophysical_audit.py`, `run_nm_mech_suite.py`, `run_nm_nature_killer_suite.py`, `run_nm_fixes_exp5_2g.py`, `run_plan2_ablations.py`).
- `manifests/`: Sanitized region-disjoint HEMIT split manifest (`split-manifest.json`) and de-identified 40-patient Orion-CRC clinical survival table (`orion_crc40_clinical_and_summary.csv`).
- `results/tables/`, `results/expansion/`, and `results/PANEL_TO_SOURCE_DATA.csv`: Complete per-unit confirmatory CSV tables across all 20 generator instances (`all20_universal_backbones.csv`, `determinism_480_parity.csv`, `hemit_*`, `orion_*`, `her2_*`, `fitted_2seed_cg_latency_scaling.csv`), expansion JSON/CSV artifacts (`comp12_all20.csv`, `orion_5fold_n40_16ch.csv`, `exp5_e2e_vs_frozen.json`, `mech_2g_throughput_scaling.json`, and mechanistic/biophysical audits), and panel-to-Source-Data crosswalk.
- `scripts/reproduce_journal_results.py`: Deterministic verification of cross-backbone wins, paired Wilcoxon test, 480-scalar determinism parity, and `N = 40` colorectal `PFS` survival concordance directly from per-unit CSV tables.
- `scripts/aggregate_nm_expansion.py`: Deterministic aggregator that builds `comp12_all20.csv`, `orion_5fold_n40_16ch.csv`, `orion_5fold_per_patient.csv`, and their summary JSONs from per-generator outputs.
- `scripts/example_tile_calibration.py`: Self-contained CPU worked example fitting a Screened Count Residual head, reloading its checkpoint, and verifying exact pre-clipping zero-DC conservation and conjugate-gradient convergence.
- `tests/`: Self-contained unit test suite verifying zero-DC gauge invariance, spectral solver accuracy, isolated false-positive count adjoint sign, Ruifrok--Johnston H-DAB channel isolation and strict cutoff boundary, duplicate-free Harrell C-index bootstrap, worked tile calibration, campaign CLI imports, and confirmatory table reproduction.

## Expansion Suite Invocation Order (GPU Reproduction)

To reproduce the full expansion campaign from raw cohort datasets and frozen generator checkpoints, run in the following order:

1. `s2.run_nm_comparators20` (12 post-hoc comparators across all 20 generator instances)
2. `s2.run_orion_5fold_16ch` (5-fold patient-stratified `N = 40` Orion 16-channel panel across all 6 generators)
3. `s2.run_nm_biophysical_audit` (tolerance radius sweep, threshold/histogram curves, spectral/CG audit, and single-cell spatial audit)
4. `s2.run_nm_mech_suite` (mechanistic suites `2A--2F`: data efficiency, feature ablation, offset/scale sweep, `lambda_dep` Pareto, cross-generator transfer, stain perturbation)
5. `s2.run_nm_nature_killer_suite` (multi-seed stability `EXP-6`, zero-shot transfer `EXP-7A`, and quantitative calibration integrity `EXP-7B`)
6. `s2.run_nm_fixes_exp5_2g` (baseline sign-verified two-seed U-Net control and initial fitted CG latency)
7. `s2.run_push1_exp5_exact_parity` (**canonical** bit-for-bit identical-objective two-architecture, five-budget `N = 32..596`, three-LR crossover suite; supersedes step 6 EXP-5)
8. `s2.run_push2_timing_spectral_tme` (**canonical** 100-distinct-tile + 1,000-tile streaming H100 latency, spectral DC/cell decomposition, 12-arm biological audit, exploratory `GlobalAffine + S_count`, and HEMIT TME ecology)
9. `s2.run_push3_orion_unseen5fold` (zero-leakage 5-fold unseen-patient end-to-end Orion-CRC training across 3 fresh fold-isolated architectures, `N = 40`)
10. `s2.run_push4_deepliif_multiorgan` (multi-organ Lung, Bladder, and Breast DeepLIIF `Ki67` single-cell and clinical Proliferation Index validation across IHC->mIF and pure Hematoxylin->mIF)
11. `s2.run_push5_her2match_full5980` (full 5,980-tile held-out HER2match audit across 3 architectures)
12. `s2.run_push6_cross_hospital_bcistainer` (cross-hospital zero-shot Beijing BCIStainer -> Dutch HER2match 5,980-tile benchmark)
13. `scripts/aggregate_nm_expansion.py` and `scripts/aggregate_nm_nature_push.py` (aggregate all expansion and multi-organ tables)

## Quick Verification (Runs on CPU in Under 15 Seconds)

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 scripts/reproduce_journal_results.py
python3 scripts/example_tile_calibration.py
```
