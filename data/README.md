# Datasets

This directory contains all data used across the project's experiments.

```
data/
├── raw/          # Input datasets (tracked in git)
├── processed/    # Intermediate benchmark results (tracked for reproducibility)
└── README.md     # This file
```

## Raw Datasets

All input CSVs are in `data/raw/`. The first four (eudirectlapse, coil2000, ausprivauto0405, freMTPL2freq_binary) form the **core 4-dataset classification benchmark suite**.

### Classification — Core Benchmark (4 datasets)

| Dataset | Label | Source | Target | Rows | Features | Pos Rate | Downloaded By |
|---------|-------|--------|--------|------|----------|----------|-------------|
| `eudirectlapse.csv` | EU Direct Lapse | [CASdatasets](https://CRAN.R-project.org/package=CASdatasets) R package (CRAN) | `lapse` | 23,060 | 18 | 12.8% | Already present |
| `coil2000.csv` | COIL 2000 (NL) | [OpenML ID 298](https://openml.org/d/298) | `CARAVAN` | 9,822 | 85 | 6.0% | `scripts/infra/download_datasets.py` |
| `ausprivauto0405.csv` | Aus. Vehicle (AU) | [CASdatasets](https://github.com/dutangc/CASdatasets) (GitHub mirror) | `ClaimOcc` | 67,856 | 7 | 6.8% | `scripts/infra/download_datasets.py` |
| `freMTPL2freq_binary.csv` | freMTPL2 Binary (FR) | Derived from freMTPL2freq.csv (below) — sampled 50K rows, binarised `ClaimNb > 0` | `ClaimIndicator` | 50,000 | 10 | 5.0% | `scripts/infra/download_datasets.py` |

Used together in: `notebooks/baseline_experiments/07_multi_dataset_benchmark.ipynb`, `scripts/legacy_finetuning/run_domain_finetune_stage_a.py`, `scripts/legacy_finetuning/evaluate_classifier_homogeneity_proposal.py`.

### Classification — Supplementary

| File | Label | Source | Target | Rows | Used In |
|------|-------|--------|--------|------|---------|
| `breast_cancer` (sklearn) | Breast Cancer (Wisconsin) | `sklearn.datasets.load_breast_cancer()` | `malignant` | 569 | Smoke/test cells in baseline notebooks |

### Regression Datasets

| File | Target(s) | Rows | Used In |
|------|-----------|------|---------|
| `freMTPL2freq.csv` | `ClaimNb` (claim frequency count) | 678,013 | Core regression benchmark: `notebooks/baseline_experiments/08_multi_dataset_regression_benchmark.ipynb` |
| `eudirectlapse.csv` | `prem_pure` (pure premium — continuous) | 23,060 | Regression benchmark notebook 08, premium target |
| `ausprivauto0405.csv` | `VehValue` (vehicle value — continuous) | 67,856 | Regression benchmark notebook 08, vehicle value target |
| `coil2000.csv` | `MINKGEM` (income — continuous) | 9,822 | Not run in notebook 08 (API limit); delivered via frontier benchmark (PR #51) |
| `freMTPL2freq_binary.csv` | `Density` (demographic — continuous) | 50,000 | Not run in notebook 08 (API limit); delivered via frontier benchmark (PR #51) |
| `freMTPL2sev.csv` (remote) | Claim severity | — | `notebooks/adswp_project/02_TabPFN_freMTPL.ipynb` — loaded from CASdatasets via pyreadr |
| `usautoBI.rda` (remote) | US Auto Bodily Injury | — | `notebooks/adswp_project/03_usautoBI_fit.ipynb` — loaded from CASdatasets via pyreadr |

### Spanish Motor Portfolio (Segura-Gisbert et al. 2024)

Two benchmark variants prepared from the raw source directory
`data/raw/Spanish motor portfolio/` (raw CSV + `Descriptive of the variables.xlsx`
data dictionary + `sample type claim.csv`) by
`scripts/infra/prepare_insurance_datasets.py --only spanish_motor`. The raw policy-year
panel is collapsed to the last policy-year per ID (bemtl16 precedent — avoids
same-policy rows leaking across random folds); `Length` NA (motorbikes) is imputed
by `Type_risk` median; exposure is treated as uniform ~1yr (no offset).

| File | Task | Target | Rows | Cols | Pos Rate | Leak Exclusions |
|------|------|--------|------|------|----------|-----------------|
| `spanish_motor_freq.csv` | Poisson frequency regression (frontier) | `N_claims_year` (int 0–18) | 53,502 | 21 | 11.1% (claims > 0) | `N_claims_history` / `R_Claims_history` (current-year leak, AUC 0.76/0.92 vs claims > 0); `Cost_claims_year` (sibling target) |
| `spanish_motor_lapse.csv` | Lapse classification (frontier) | `LapseB` = (`Lapse` > 0) | 53,502 | 23 | 35.4% | `Date_lapse` (termination date, presence AUC 0.85 vs lapse) |

Source: Segura-Gisbert, M., et al. (2024). "Dataset of an actual motor vehicle
insurance portfolio." Rebuild with: `python scripts/infra/prepare_insurance_datasets.py --only spanish_motor`.

### Sklearn Demo Datasets

| Dataset | Used In |
|---------|---------|
| `load_breast_cancer()` | Embedding workflow, baseline smoke tests |
| `load_diabetes()` | `notebooks/adswp_project/04_tabpfn_embedding_workflow.ipynb` |

## Data Provenance

### CASdatasets (R package)
The R package `CASdatasets` (Christophe Dutang) is the primary source for insurance data:

- **Install:** `install.packages('CASdatasets')` or from [CRAN](https://CRAN.R-project.org/package=CASdatasets)
- **GitHub mirror:** `https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/`
- **Loader:** `pyreadr.read_r()` for `.rda` files
- **CSV copies:** `eudirectlapse.csv`, `ausprivauto0405.csv` (via download script), `freMTPL2freq.csv`

### OpenML
- **coil2000:** `sklearn.datasets.fetch_openml('coil2000', version=2)`

### Derived In-House
- **freMTPL2freq_binary:** Sampled from `freMTPL2freq.csv` (50K of 678K rows), target `ClaimIndicator = ClaimNb > 0`

## Attribution

- **CASdatasets:** Dutang, C., & Charpentier, A. (2020). CASdatasets: Insurance datasets. R package version 1.0-11. https://CRAN.R-project.org/package=CASdatasets
- **COIL 2000:** OpenML dataset ID 298. Van der Putten, P., & Van Someren, M. (2000). CoIL Challenge 2000.
- **freMTPL2freq:** French motor third-party liability insurance data, from CASdatasets.
