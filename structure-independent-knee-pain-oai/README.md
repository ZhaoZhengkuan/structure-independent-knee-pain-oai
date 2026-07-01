# Structure-Independent Knee Pain as a Harbinger of Future Joint Damage

Analysis code for the prospective study of the symptom–structure gap in the Osteoarthritis Initiative (OAI). The pipeline derives a structure-independent knee-pain residual, tests whether it forecasts structural progression across three endpoints, triangulates its molecular/imaging/psychosocial correlates with formal mediation, and quantifies its independent value for predicting total knee replacement.

## Repository contents

| File | Purpose |
|---|---|
| `00_extract_longitudinal.py` | Extracts and harmonises baseline and longitudinal variables from the OAI source tables (WOMAC pain, KL grade, quantitative mJSW, femorotibial angle, longitudinal mJSW/KL, MOAKS, FNIH biomarkers, TKR outcomes). |
| `01_progression_mixed_model.py` | Constructs the structure-independent residual (robust regression, five specifications) and fits the primary linear mixed-effects model for the residual-by-time interaction on medial mJSW (the harbinger decision gate). |
| `02_mechanism_triangulation.py` | Correlates the residual with FNIH cartilage/bone turnover biomarkers (FDR-controlled), MOAKS inflammatory features, and central/affective markers. |
| `03_tkr_prognosis.py` | Nested cause-specific Cox models, repeated cross-validated discrimination/calibration, reclassification metrics, and SIMEX for the total knee replacement endpoint. |
| `04_replication_robustness_mediation.py` | Mediation of the residual–progression association through baseline inflammation; replication on KL-grade worsening and the ≥0.7 mm progressor endpoint; sensitivity battery (index knee, KL≥2, censoring at TKR, attrition IPW); descriptive Table 1; decision-curve net-benefit computation. |

Figure-generation scripts are intentionally excluded from this repository; all statistical outputs are written as CSV tables.

## Requirements

Python ≥ 3.10. Install dependencies with:

```bash
pip install -r requirements.txt
```

Core libraries: pandas, numpy, scipy, statsmodels, lifelines, scikit-learn.

## Data access

This analysis uses data from the Osteoarthritis Initiative (OAI), a public–private partnership. **OAI source data are not redistributed here.** Investigators may obtain the data after registration and acceptance of the OAI Data Use Agreement via the OAI repository (https://nda.nih.gov/oai/). Place the extracted analysis datasets in a `data/` directory at the repository root; the scripts expect the derived files produced by `00_extract_longitudinal.py` (e.g., `c2_derived.csv` and the longitudinal/MOAKS/FNIH extracts).

## Reproducing the analysis

```bash
# 1. Extract and harmonise OAI variables (requires OAI data in ./data)
python 00_extract_longitudinal.py

# 2. Primary harbinger model (residual construction + mixed-effects decision gate)
python 01_progression_mixed_model.py

# 3. Mechanistic triangulation
python 02_mechanism_triangulation.py

# 4. Total knee replacement prognosis
python 03_tkr_prognosis.py

# 5. Replication, robustness, mediation, Table 1, decision-curve data
python 04_replication_robustness_mediation.py
```

Each stage writes its results to a dedicated output directory as CSV.

## Statistical summary

- **Primary endpoint:** rate of medial mJSW loss over 96 months; linear mixed-effects model with random intercepts and slopes nested within knee; the residual-by-time interaction is the adjudicating quantity.
- **Confirmatory endpoints:** ordinal KL-grade worsening and a ≥0.7 mm joint-space-loss progressor endpoint (cause-specific Cox models).
- **Mechanism:** Pearson correlations with FDR control; mediation by difference-in-coefficients with indirect-effect estimation.
- **Prognosis:** nested cause-specific Cox models, repeated five-fold cross-validation, calibration, decision-curve analysis, and SIMEX.

## License

Code is released under the MIT License (see `LICENSE`). Use of OAI data is governed separately by the OAI Data Use Agreement.

## Citation

If you use this code, please cite the associated manuscript (details to be added upon publication).
