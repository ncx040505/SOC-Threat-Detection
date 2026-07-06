#!/usr/bin/env python3
"""
Fast Baseline: LightGBM + CV on sampled data → res.csv
Minimal overhead, one-pass, no heavy IP features.
"""

import gc, logging, os, sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# ===== 1. Load =====
logger.info("Loading data...")
t0 = time.time()
train = pd.read_parquet(DATA / "train.parquet")
test = pd.read_parquet(DATA / "valid_input.parquet")
logger.info("Train: %s, Test: %s (%.1fs)", train.shape, test.shape, time.time() - t0)

# ===== 2. Label =====
label_map = {"benign": 0, "malicious": 1, "suspicious": 2}
y = train["label_binary"].map(label_map).values.astype(np.int32)
n_classes = len(label_map)
logger.info("Label dist: benign=%d, malicious=%d, suspicious=%d",
            (y==0).sum(), (y==1).sum(), (y==2).sum())

# ===== 3. Build features (train + test together) =====
logger.info("Building features...")
meta_cols = {"event_id", "message_sanitized", "label_binary", "label"}

# --- Numeric features ---
num_cols = [c for c in train.columns
            if c not in meta_cols and c != "label_binary"
            and pd.api.types.is_numeric_dtype(train[c])]
logger.info("Numeric cols: %s", num_cols)

# Combine for scaling
n_train = len(train)
all_num = pd.concat([train[num_cols], test[num_cols]], axis=0, ignore_index=True)

# 简单均值填充
for c in num_cols:
    m = all_num[c].median()
    all_num[c] = all_num[c].fillna(m)

scaler = RobustScaler()
X_num = scaler.fit_transform(all_num)
X_num_train = X_num[:n_train]
X_num_test = X_num[n_train:]
del all_num, X_num; gc.collect()

# --- Categorical → Label Encode ---
cat_cols = [c for c in train.columns
            if c not in meta_cols and not pd.api.types.is_numeric_dtype(train[c])]
logger.info("Cat cols: %s", cat_cols)

X_cat_train = np.zeros((n_train, len(cat_cols)), dtype=np.float32)
X_cat_test = np.zeros((len(test), len(cat_cols)), dtype=np.float32)

for i, c in enumerate(cat_cols):
    le = LabelEncoder()
    all_cat = pd.concat([train[c].fillna("__NULL__"), test[c].fillna("__NULL__")], axis=0)
    encoded = le.fit_transform(all_cat).astype(np.float32)
    X_cat_train[:, i] = encoded[:n_train]
    X_cat_test[:, i] = encoded[n_train:]
    logger.info("  Encoded '%s': %d uniques", c, len(le.classes_))

# --- Text TF-IDF ---
logger.info("TF-IDF on message_sanitized...")
all_text = pd.concat([train["message_sanitized"].fillna(""), test["message_sanitized"].fillna("")], axis=0)

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    analyzer='char_wb',
    min_df=5,
    max_df=0.8,
    sublinear_tf=True,
    dtype=np.float32,
)
X_tfidf = tfidf.fit_transform(all_text)
X_tfidf_train = X_tfidf[:n_train]
X_tfidf_test = X_tfidf[n_train:]
del all_text; gc.collect()
logger.info("TF-IDF shape: %s", X_tfidf.shape)

# ===== 4. Assemble feature matrix =====
logger.info("Assembling features...")
X_train = hstack([
    csr_matrix(X_num_train),
    csr_matrix(X_cat_train),
    X_tfidf_train,
]).tocsr()
X_test = hstack([
    csr_matrix(X_num_test),
    csr_matrix(X_cat_test),
    X_tfidf_test,
]).tocsr()
del X_num_train, X_num_test, X_cat_train, X_cat_test, X_tfidf_train, X_tfidf_test
gc.collect()
logger.info("Train features: %s, Test features: %s", X_train.shape, X_test.shape)

# ===== 5. CV + Predict =====
logger.info("Running 5-fold CV with LightGBM...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": 6,
    "verbose": -1,
    "device": "cpu",
}

all_preds = np.zeros((len(test), n_classes), dtype=np.float64)
f1_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    logger.info("Fold %d/5", fold + 1)
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(100),
            log_evaluation(500),
        ],
    )

    val_pred = model.predict(X_val)
    f1 = f1_score(y_val, val_pred, average="macro")
    f1_scores.append(f1)
    logger.info("  Fold %d Macro F1: %.4f", fold + 1, f1)

    # Predict test
    pred_proba = model.predict_proba(X_test)
    all_preds += pred_proba / skf.n_splits

    del model, X_tr, X_val, y_tr, y_val; gc.collect()

logger.info("CV Macro F1: %.4f ± %.4f", np.mean(f1_scores), np.std(f1_scores))

# ===== 6. Output res.csv =====
logger.info("Generating res.csv...")
pred_labels = np.argmax(all_preds, axis=1)
reverse_map = {0: "benign", 1: "malicious", 2: "suspicious"}
pred_strings = [reverse_map[l] for l in pred_labels]

sub = pd.DataFrame({
    "event_id": test["event_id"],
    "label_binary": pred_strings,
})
os.makedirs(OUT, exist_ok=True)
out_path = OUT / "res.csv"
sub.to_csv(out_path, index=False)
logger.info("Saved to %s (%d rows)", out_path, len(sub))

# Quick stats
logger.info("Prediction distribution:\n%s", sub["label_binary"].value_counts().to_string())

logger.info("Done!")
