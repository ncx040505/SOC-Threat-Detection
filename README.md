# SOC Threat Detection

Multi-model ensemble for SOC log threat detection, classifying security events into **benign**, **malicious**, and **suspicious** categories.

## Models

| Model | Accuracy | F1 (macro) | AUC (OvR) | Train Time |
|---|---|---|---|---|
| Random Forest | 0.99999 | 0.99990 | 0.99999 | 85.9s |
| XGBoost | 0.99999 | 0.99992 | 1.00000 | 6.8s |
| LightGBM | 0.99999 | 0.99992 | 1.00000 | 23.0s |
| MLP | 0.99999 | 0.99995 | 0.99999 | 184.6s |
| Voting Ensemble | 1.00000 | 1.00000 | 1.00000 | — |
| Stacking | 1.00000 | 1.00000 | 1.00000 | — |

## Architecture

```
data/           ← raw data (not included)
│
├── src/
│   ├── config_loader.py     ← YAML config parsing
│   ├── data_loader.py       ← Parquet loading, tokenization
│   ├── feature_engineer.py  ← Feature engineering pipeline
│   ├── ip_features.py       ← IP address feature extraction
│   ├── imbalance_handler.py ← Hybrid under/over sampling
│   ├── models.py            ← RF, XGBoost, LightGBM, MLP, ensembles
│   ├── train.py             ← Training orchestration
│   ├── predict.py           ← Inference
│   ├── explainer.py         ← SHAP explanations
│   └── visualizer.py        ← Confusion matrices, ROC, etc.
├── configs/config.yaml      ← Configuration
├── tests/                   ← Pytest unit tests
├── models/                  ← Trained .pkl artifacts
└── outputs/                 ← Predictions & metrics
```

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Training
python src/train.py

# Prediction
python src/predict.py --input data/valid_input.parquet --output outputs/res.csv
```

## Feature Engineering

- **Token-level TF-IDF** over sanitized log messages (char-wb n-grams)
- **IP address features** (octets, subnet classification, entropy)
- **Interaction features** (top-k feature crosses)
- **Hybrid imbalance handling**: undersampling class 0 + SMOTE oversampling classes 1 & 2

## v2.1 — Cross-Environment Diagnosis & Fix

Key finding: the holdout/test logs come from a **different host environment** (`src_host` 100% OOV, product distribution shifted). Same-distribution validation metrics (F1 ≈ 1.0) were inflated and masked this.

**Diagnosis**

- Product ↔ label purity: `Precinct` / `Falcon` / `AWS VPC Security` / `ASA Firewall` ≈ 100% suspicious; all malicious events come from `syslog` + empty `product_name`
- Baseline model judged all 41k pure-product samples as benign on the test set (train control: 99.96% correct)

**Fixes**

1. **Product-purity rule layer** — pure-product samples → `suspicious` (removes the heaviest penalty path: threat→benign)
2. **Hard content features** (`is_sus_product`, `is_empty_product`, `username_is_dash`, `has_src_ip`, `src_ip_cgnat`, `msg_flow_tokens`) — content-level signals survive host OOV; empty-product malicious recall → 1.0
3. **Two-stage pipeline** (`models/two_stage_v3.pkl`) — empty-product subset → content model (mal vs benign); pure-product → rule; rest → benign
4. **Cross-host validation protocol** — grouped split on `src_host` (true OOD estimate): macro F1 0.6594 → **0.7178**; submission validated by two independent methods (ensemble + content model, Jaccard 0.996)

**Artifacts**: `outputs/res_final.csv` (final submission), `models/two_stage_v3.pkl`, `models/lgbm_hardfeat.pkl`

## License

MIT
