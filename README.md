# Screened Count Residual: Spatial Threshold Calibration for Virtual Immunostaining

Standalone, CPU-verifiable reference implementation (`v2.0.0`), worked calibration example, split manifests, and audited per-unit result tables for gauge-fixed Screened Count Residuals across 20 pretrained virtual-staining generator instances and six cohorts spanning five human and murine organ systems (`HEMIT`, `Harvard Orion-CRC`, `HER2match`, `MIST` four-IHC, `DeepLIIF` Lung/Bladder/Breast, and `Mouse Brain and Dura Mater`).

## Repository Layout

- `s2/`: Complete transitive import closure of core operators (`query_relation_reconstruction.py`, `query_relation_count_objective.py`), plug-and-play universal residual head (`screened_universal_head.py`), generator adapters (`backbone_zoo.py`), single-cell spatial proximity and duplicate-free time-to-recurrence survival evaluator (`clinical_cell_survival.py`), confirmatory campaign runners (`run_hemit_multibackbone_cell.py`, `run_orion_multibackbone_survival.py`, `run_her2match_dab_campaign.py`), expansion runners (`run_nm_*.py`, `run_push*.py`), and `v2.0.0` flagship runners (`run_nv2_n1_*.py` through `run_nv2_n8_brain_dura_labelfree.py`).
- `manifests/`: Sanitized region-disjoint HEMIT split manifest (`split-manifest.json`), patient-disjoint Orion 5-fold manifest (`orion-patient-cv-manifest.json`), and de-identified 40-patient Orion-CRC clinical survival table (`orion_crc40_clinical_and_summary.csv`).
- `results/tables/`, `results/expansion/`, `results/nm_nature_push/`, `results/nature_v2/`, and `results/PANEL_TO_SOURCE_DATA.csv`: Complete per-unit confirmatory CSV tables across all 20 generator instances (`all20_universal_backbones.csv`, `determinism_480_parity.csv`), unified full-tile Orion survival (`nv2_n1_orion_survival_unified.csv`, `nv2_n1_orion_patient_scores_unified.csv`), 18-arm comparator matrix (`nv2_n3_comp18_all20.csv`), mechanistic XAI (`nv2_n2_mechanistic_xai.json`), mosaic seam continuity (`nv2_n4_wsi_seam_composition.json`), target-site Beijing-to-Dutch transfer (`nv2_n5_cross_hospital_pyramidpix2pix_her2match_dab1.json`), ultra-low-shot (`nv2_n6_ultralowshot.json`), slide-disjoint MIST four-IHC (`nv2_n7_mist_*_dab1.json`), mouse brain and dura mater (`nv2_n8_brain_dura_labelfree.json`, `nv2_n8_brain_dura_partB_pbs_lps.csv`), and the 268-panel Source Data crosswalk.
- `scripts/reproduce_journal_results.py`: Deterministic CPU verification of all 20-generator confirmatory comparisons, 18-arm comparator frontier checks, 480-scalar determinism parity, zero-leakage 15-fold unseen Orion provenance, unified full-tile `N = 40` time-to-recurrence survival checks, and 268-panel mapping directly from archived tables.
- `scripts/example_tile_calibration.py`: Self-contained CPU worked example fitting a Screened Count Residual head, reloading its checkpoint, and verifying exact pre-clipping zero-DC conservation and conjugate-gradient convergence.
- `tests/`: Self-contained unit test suite verifying zero-DC gauge invariance, spectral solver accuracy, isolated false-positive count adjoint sign, Ruifrok--Johnston H-DAB channel isolation and strict cutoff boundary, duplicate-free Harrell C-index bootstrap, worked tile calibration, campaign CLI imports, and confirmatory table reproduction.

## Quick Verification (Runs on CPU in Under 15 Seconds)

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 scripts/reproduce_journal_results.py
python3 scripts/example_tile_calibration.py
```
