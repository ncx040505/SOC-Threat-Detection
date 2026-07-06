"""
快速实验: IP 行为特征 + LightGBM 基线
======================================
只跑 LightGBM + CV，加入 IP 行为统计 + 拓扑特征
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ip_features import IPBehaviorFeatures, IPTopologyFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 数据加载
# ============================================================
logger.info("加载数据...")
t0 = time.time()
train = pd.read_parquet(ROOT / "data/train.parquet")
test = pd.read_parquet(ROOT / "data/valid_input.parquet")
logger.info("Train: %d 行, Test: %d 行 (%.1fs)", len(train), len(test), time.time() - t0)

# ============================================================
# 标签处理
# ============================================================
label_enc = LabelEncoder()
y = label_enc.fit_transform(train["label_binary"])
logger.info("标签: %s → %s", label_enc.classes_, np.arange(len(label_enc.classes_)))

# ============================================================
# 基础数值特征
# ============================================================
logger.info("提取基础数值特征...")
meta_cols = {"event_id", "message_sanitized", "label_binary", "label"}

numeric_cols = [c for c in train.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in train.columns if c not in meta_cols and c not in numeric_cols]
logger.info("数值列: %d, 类别列: %d", len(numeric_cols), len(cat_cols))

# 填充 + scale
for c in numeric_cols:
    med = train[c].median()
    train[c] = train[c].fillna(med)
    test[c] = test[c].fillna(med)

scaler = RobustScaler()
X_num_tr = scaler.fit_transform(train[numeric_cols].values.astype(np.float32))
X_num_te = scaler.transform(test[numeric_cols].values.astype(np.float32))

# 类别列 → label encoding
cat_tr = np.zeros((len(train), len(cat_cols)), dtype=np.float32)
cat_te = np.zeros((len(test), len(cat_cols)), dtype=np.float32)
for i, c in enumerate(cat_cols):
    le = LabelEncoder()
    combined = pd.concat([train[c].fillna("__MISSING__"), test[c].fillna("__MISSING__")])
    le.fit(combined)
    cat_tr[:, i] = le.transform(train[c].fillna("__MISSING__"))
    cat_te[:, i] = le.transform(test[c].fillna("__MISSING__"))

# ============================================================
# TF-IDF
# ============================================================
logger.info("TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 4), analyzer="char_wb",
    min_df=3, sublinear_tf=True, dtype=np.float32,
)
X_tfidf_tr = tfidf.fit_transform(train["message_sanitized"].fillna(""))
X_tfidf_te = tfidf.transform(test["message_sanitized"].fillna(""))
logger.info("TF-IDF: %d features", X_tfidf_tr.shape[1])

# ============================================================
# IP 特征
# ============================================================
logger.info("IP 行为特征...")
ip_bf = IPBehaviorFeatures()
ip_bf.fit(train)
X_ip_bf_tr = ip_bf.transform(train)
X_ip_bf_te = ip_bf.transform(test)

logger.info("IP 拓扑特征...")
ip_tf = IPTopologyFeatures()
ip_tf.fit(train)
X_ip_topo_tr = ip_tf.transform(train)
X_ip_topo_te = ip_tf.transform(test)

# ============================================================
# 拼接
# ============================================================
logger.info("拼接...")
X_base_tr = np.hstack([X_num_tr, cat_tr, X_ip_bf_tr, X_ip_topo_tr])
X_base_te = np.hstack([X_num_te, cat_te, X_ip_bf_te, X_ip_topo_te])

X_all_tr = hstack([X_base_tr, X_tfidf_tr]).tocsr()
X_all_te = hstack([X_base_te, X_tfidf_te]).tocsr()
logger.info("最终特征: %d", X_all_tr.shape[1])

del X_num_tr, X_num_te, cat_tr, cat_te, X_ip_bf_tr, X_ip_bf_te, X_ip_topo_tr, X_ip_topo_te, X_base_tr, X_base_te
gc.collect()

# ============================================================
# LightGBM CV
# ============================================================
logger.info("LightGBM 5-fold CV...")
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=10, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", metric="multi_logloss",
    boosting_type="gbdt", random_state=42, n_jobs=-1, verbose=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
t_fit = time.time()
for fold, (tr_i, va_i) in enumerate(skf.split(X_all_tr, y)):
    lgbm.fit(X_all_tr[tr_i], y[tr_i])
    pred = lgbm.predict(X_all_tr[va_i])
    f1 = f1_score(y[va_i], pred, average="macro")
    scores.append(f1)
    logger.info("  Fold %d: macro F1 = %.5f", fold + 1, f1)

logger.info("CV macro F1: %.5f ± %.5f (%.1fs)", np.mean(scores), np.std(scores), time.time() - t_fit)

# ============================================================
# 全量训练 → 预测
# ============================================================
logger.info("全量训练 + 预测...")
lgbm.fit(X_all_tr, y)
pred_te = lgbm.predict(X_all_te)
labels_te = label_enc.inverse_transform(pred_te)

sub = pd.DataFrame({"UID": test["message_sanitized"], "label": labels_te})
out = ROOT / "outputs/submission_ip_features.csv"
out.parent.mkdir(exist_ok=True)
sub.to_csv(out, index=False)
logger.info("输出: %s (%d rows)", out, len(sub))

for lbl, cnt in sub["label"].value_counts().items():
    logger.info("  %s: %d (%.1f%%)", lbl, cnt, cnt / len(sub) * 100)

logger.info("✅ 完成")
