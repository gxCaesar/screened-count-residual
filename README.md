# Screened Count Residual: Spatial Threshold Calibration for Virtual Immunostaining

Standalone, CPU-verifiable reference implementation, worked calibration example, split manifests, and confirmatory per-unit result tables for gauge-fixed Screened Count Residuals across 20 pretrained virtual-staining generator instances spanning three cohorts (`HEMIT`, `Harvard Orion CRC`, and `HER2match`).

## Repository Layout

- `s2/`: Complete transitive import closure of core operators (`query_relation_reconstruction.py`, `query_relation_count_objective.py`), plug-and-play universal residual head (`screened_universal_head.py`), generator adapters (`backbone_zoo.py`), single-cell spatial proximity and duplicate-free survival evaluator (`clinical_cell_survival.py`), and confirmatory campaign runners (`run_hemit_multibackbone_cell.py`, `run_orion_multibackbone_survival.py`, `run_her2match_dab_campaign.py`).
- `manifests/`: Sanitized region-disjoint HEMIT split manifest (`split-manifest.json`) and de-identified 40-patient Orion-CRC clinical survival table (`orion_crc40_clinical_and_summary.csv`).
- `results/tables/` and `results/PANEL_TO_SOURCE_DATA.csv`: Complete per-unit confirmatory CSV tables across all 20 generator instances (`all20_universal_backbones.csv`, `determinism_480_parity.csv`, `hemit_*`, `orion_*`, `her2_*`, `inference_latency_and_clip_audit.csv`) and panel-to-Source-Data crosswalk.
- `scripts/reproduce_journal_results.py`: Deterministic verification of cross-backbone wins, paired Wilcoxon test, 480-scalar determinism parity, and `N = 40` colorectal `PFS` survival concordance directly from per-unit CSV tables.
- `scripts/example_tile_calibration.py`: Self-contained CPU worked example fitting a Screened Count Residual head, reloading its checkpoint, and verifying exact pre-clipping zero-DC conservation and conjugate-gradient convergence.
- `tests/`: Self-contained unit test suite verifying zero-DC gauge invariance, spectral solver accuracy, isolated false-positive count adjoint sign, Ruifrok--Johnston H-DAB channel isolation and strict cutoff boundary, duplicate-free Harrell C-index bootstrap, worked tile calibration, campaign CLI imports, and confirmatory table reproduction.

## Quick Verification (Runs on CPU in Under 15 Seconds)

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 scripts/reproduce_journal_results.py
python3 scripts/example_tile_calibration.py
```
