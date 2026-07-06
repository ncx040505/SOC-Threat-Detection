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
python src/predict.py --input data/valid_input.parquet --output outputs/pred.csv
```

## Feature Engineering

- **Token-level TF-IDF** over sanitized log messages (char-wb n-grams)
- **IP address features** (octets, subnet classification, entropy)
- **Interaction features** (top-k feature crosses)
- **Hybrid imbalance handling**: undersampling class 0 + SMOTE oversampling classes 1 & 2

## License

MIT
