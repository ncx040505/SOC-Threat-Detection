"""
特征工程模块
✅ v2.0 重大改进:
  - 修复 P0-#5: 交互特征 fit 时记录列名列表，transform 严格复用
  - 修复 P2-#2: 每个子组件独立拟合状态追踪
  - 新增 P1-#8: 特征选择 (SelectFromModel / Mutual Info / TruncatedSVD)
  - 列名从 config.yaml columns 读取
  - 版本管理 (__getstate__ / __setstate__)
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder

from src.config_loader import Config

logger = logging.getLogger(__name__)

# 可选
try:
    from category_encoders import TargetEncoder
    _HAS_TE = True
except ImportError:
    _HAS_TE = False
    TargetEncoder = None
    logger.warning("category_encoders 未安装，Target Encoding 将回退为 Label Encoding")


class FitState:
    """单个组件的拟合状态"""

    def __init__(self, name: str):
        self.name = name
        self.fitted = False
        self.transformer: Any = None
        self.column_names_before: List[str] = []
        self.column_names_after: List[str] = []
        self.meta: Dict[str, Any] = {}

    def mark_fitted(self):
        self.fitted = True


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """特征工程管道"""

    def __init__(self, config: Config):
        self.cfg = config
        self._version: int = 2

        # ---- 列名从 config 读取 ----
        self.id_cols = list(config.get("columns", {}).get("id_cols", ["event_id", "raw_message"]))
        self.label_col = config.get("columns", {}).get("label_col", "label")
        self.text_col = config.get("columns", {}).get("text_col", "raw_message")

        # ---- 子组件状态 ----
        self.states: Dict[str, FitState] = {
            "scaler": FitState("scaler"),
            "target_encoder": FitState("target_encoder"),
            "tfidf": FitState("tfidf"),
            "interactions": FitState("interactions"),
            "feature_selector": FitState("feature_selector"),
            "svd": FitState("svd"),
        }

        # ---- 列分类结果 ----
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.high_cardinality_cols: List[str] = []
        self.low_cardinality_cols: List[str] = []
        self._column_classified = False

    # ================================================================
    # 版本管理
    # ================================================================
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_version"] = 2
        return state

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)
        if state.get("_version", 1) < 2:
            logger.warning("加载旧版本 FeatureEngineer (v%d)，兼容模式", state.get("_version", 1))

    # ================================================================
    # fit
    # ================================================================
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """完整拟合所有子组件"""
        logger.info("FeatureEngineer fit — 输入 shape=%s", X.shape)

        if not self._column_classified:
            self._classify_columns(X)

        current_X = X.copy()

        # ---- 1. 数值 Scaler ----
        if self.numeric_cols:
            numeric_data = current_X[self.numeric_cols].copy()
            if self.cfg.get("features", {}).get("numeric_strategy", "robust") == "standard":
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            scaler.fit(numeric_data)
            self.states["scaler"].transformer = scaler
            self.states["scaler"].column_names_before = list(self.numeric_cols)
            self.states["scaler"].column_names_after = list(self.numeric_cols)
            self.states["scaler"].mark_fitted()
            logger.info("Scaler 已拟合 (%s), %d列", type(scaler).__name__, len(self.numeric_cols))

        # ---- 2. 类别 Target Encoding ----
        te_cfg = self.cfg.get("features", {}).get("target_encoding", {})
        if te_cfg.get("enabled", True) and self.high_cardinality_cols and y is not None:
            te_cols = te_cfg.get("cols", [])
            if not te_cols:
                te_cols = self.high_cardinality_cols
            else:
                te_cols = [c for c in te_cols if c in current_X.columns]

            if te_cols:
                if _HAS_TE and TargetEncoder is not None:
                    smooth = te_cfg.get("smooth", 10)
                    te = TargetEncoder(cols=te_cols, smoothing=smooth)
                    te.fit(current_X[te_cols], y)
                    self.states["target_encoder"].transformer = te
                    self.states["target_encoder"].column_names_before = list(te_cols)
                    self.states["target_encoder"].column_names_after = list(te_cols)
                    self.states["target_encoder"].mark_fitted()
                    logger.info("TargetEncoder 已拟合, %d列", len(te_cols))
                else:
                    # 回退：Label Encoding
                    from sklearn.preprocessing import OrdinalEncoder
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    enc.fit(current_X[te_cols].astype(str))
                    self.states["target_encoder"].transformer = enc
                    self.states["target_encoder"].meta["fallback"] = "ordinal"
                    self.states["target_encoder"].column_names_before = list(te_cols)
                    self.states["target_encoder"].column_names_after = list(te_cols)
                    self.states["target_encoder"].mark_fitted()
                    logger.info("TargetEncoder 回退为 OrdinalEncoder, %d列", len(te_cols))

        # ---- 3. 低基数类别 Label Encoding (fit 时记录映射) ----
        if self.low_cardinality_cols:
            le_maps = {}
            for col in self.low_cardinality_cols:
                if col in current_X.columns:
                    le = LabelEncoder()
                    le.fit(current_X[col].astype(str).fillna("UNKNOWN"))
                    le_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            self.states["target_encoder"].meta["low_card_le_maps"] = le_maps

        # ---- 4. TF-IDF ----
        tf_cfg = self.cfg.get("features", {}).get("text_feature", {})
        if tf_cfg.get("enabled", True) and self.text_col in current_X.columns:
            text_data = current_X[self.text_col].fillna("").astype(str)
            if text_data.str.len().sum() > 0:
                tfidf = TfidfVectorizer(
                    max_features=tf_cfg.get("max_features", 5000),
                    ngram_range=tuple(tf_cfg.get("ngram_range", [1, 4])),
                    analyzer=tf_cfg.get("analyzer", "char_wb"),
                    min_df=tf_cfg.get("min_df", 3),
                    sublinear_tf=True,
                )
                tfidf.fit(text_data)
                self.states["tfidf"].transformer = tfidf
                self.states["tfidf"].column_names_before = [self.text_col]
                self.states["tfidf"].column_names_after = [
                    f"tfidf_{i}" for i in range(len(tfidf.get_feature_names_out()))
                ]
                self.states["tfidf"].mark_fitted()
                logger.info("TF-IDF 已拟合, %d 特征", len(tfidf.get_feature_names_out()))

        # ---- 5. 交互特征 — fit 时只记录列名（实际计算在 transform 中） ----
        if self.cfg.get("features", {}).get("interaction_features", True):
            # 用一小部分数据计算交互特征方差，选出 top-k
            sample_X = current_X.head(min(5000, len(current_X)))
            interactions = self._compute_interactions(sample_X)
            variances = interactions.var().sort_values(ascending=False)
            top_k = self.cfg.get("features", {}).get("interaction_top_k", 100)
            selected = variances.head(top_k).index.tolist()
            self.states["interactions"].meta["selected_columns"] = selected
            self.states["interactions"].column_names_before = []
            self.states["interactions"].column_names_after = selected
            self.states["interactions"].mark_fitted()
            logger.info("交互特征已记录, 选 %d 列", len(selected))

        # ---- 6. 特征选择 — 用全量数据拟合 ----
        sel_cfg = self.cfg.get("features", {}).get("feature_selection", {})
        if sel_cfg.get("enabled", True):
            X_all, _ = self._apply_transforms(current_X)
            y_np = y.values if hasattr(y, "values") else np.asarray(y)

            method = sel_cfg.get("method", "select_from_model")
            max_feat = sel_cfg.get("max_features", 300)

            if method == "select_from_model":
                model_type = sel_cfg.get("model", "random_forest")
                threshold = sel_cfg.get("threshold", "median")

                if model_type == "random_forest":
                    sel_model = RandomForestClassifier(
                        n_estimators=50, max_depth=10,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    )
                else:
                    sel_model = RandomForestClassifier(
                        n_estimators=50, max_depth=10,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    )

                sel_model.fit(X_all, y_np)
                selector = SelectFromModel(sel_model, prefit=True, threshold=threshold, max_features=max_feat)
                selected_mask = selector.get_support()
                selected_cols = X_all.columns[selected_mask].tolist()

            elif method == "mutual_info":
                mi = mutual_info_classif(X_all, y_np, random_state=42)
                mi_series = pd.Series(mi, index=X_all.columns).sort_values(ascending=False)
                selected_cols = mi_series.head(max_feat).index.tolist()

            elif method == "pca":
                from sklearn.decomposition import PCA
                n_comp = min(max_feat, X_all.shape[1], X_all.shape[0] - 1)
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_all)
                self.states["svd"].transformer = pca
                self.states["svd"].column_names_before = list(X_all.columns)
                self.states["svd"].column_names_after = [f"pca_{i}" for i in range(n_comp)]
                self.states["svd"].mark_fitted()
                selected_cols = []  # PCA 不产生列名选择，走 svd 路径

            else:
                selected_cols = []

            if selected_cols:
                self.states["feature_selector"].meta["selected_columns"] = selected_cols
                self.states["feature_selector"].column_names_before = list(X_all.columns)
                self.states["feature_selector"].column_names_after = selected_cols
                self.states["feature_selector"].mark_fitted()
                logger.info("特征选择完成: %d/%d 列 (%s)", len(selected_cols), X_all.shape[1], method)

        # ---- 计算最终特征名 ----
        self._finalize_feature_names()

        logger.info("FeatureEngineer fit 完成 — 最终特征数: %d", len(self.all_feature_names))
        return self

    # ================================================================
    # transform
    # ================================================================
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, event_ids: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        转换特征。
        返回: (X_transformed, y_raw, event_id_series)
        """
        if not self._any_state_fitted():
            raise RuntimeError("FeatureEngineer 尚未 fit！")

        logger.info("FeatureEngineer transform — 输入 shape=%s", X.shape)

        df = X.copy()

        # 提取 event_id 和 label（优先使用外部传入的 event_ids）
        if event_ids is None:
            for c in self.id_cols:
                if c in df.columns and c != self.text_col:
                    event_ids = df[c].copy()
                    break

        y_result = None
        if y is not None:
            y_result = y
        elif self.label_col in df.columns:
            y_result = df[self.label_col]

        # ---- 1. Scaler ----
        state = self.states["scaler"]
        if state.fitted and state.transformer is not None:
            fit_cols = list(state.column_names_before)
            available = [c for c in fit_cols if c in df.columns]
            if available:
                X_scaler = df[available].copy()
                for c in fit_cols:
                    if c not in X_scaler.columns:
                        X_scaler[c] = 0
                X_scaler = X_scaler[fit_cols]
                scaled = state.transformer.transform(X_scaler)
                df_scaled = pd.DataFrame(scaled, columns=fit_cols, index=df.index)
                for c in fit_cols:
                    df[c] = df_scaled[c]

        # ---- 2. Target Encoding ----
        state = self.states["target_encoder"]
        if state.fitted and state.transformer is not None:
            cols = [c for c in state.column_names_before if c in df.columns]
            if cols:
                if state.meta.get("fallback") == "ordinal":
                    encoded = state.transformer.transform(df[cols].astype(str))
                    df[cols] = pd.DataFrame(encoded, columns=cols, index=df.index)
                else:
                    df[cols] = state.transformer.transform(df[cols])

        # ---- 3. 低基数 Label Encoding ----
        le_maps = self.states["target_encoder"].meta.get("low_card_le_maps", {})
        for col, mapping in le_maps.items():
            if col in df.columns:
                default_val = -1
                df[col] = df[col].astype(str).fillna("UNKNOWN").map(mapping).fillna(default_val).astype(int)

        # ---- 4. TF-IDF ----
        state = self.states["tfidf"]
        tfidf_features = None
        if state.fitted and state.transformer is not None and self.text_col in df.columns:
            text_data = df[self.text_col].fillna("").astype(str)
            tfidf_matrix = state.transformer.transform(text_data)
            if sp_sparse.issparse(tfidf_matrix):
                tfidf_matrix = tfidf_matrix.toarray()
            tfidf_features = pd.DataFrame(
                tfidf_matrix,
                columns=state.column_names_after,
                index=df.index,
                dtype=np.float32,
            )
            # 移除原始 text 列
            if self.text_col in df.columns:
                df = df.drop(columns=[self.text_col])

        # ---- 5. 交互特征 (严格用 fit 时记录的列名) ----
        state = self.states["interactions"]
        interaction_features = None
        if state.fitted and state.meta.get("selected_columns"):
            selected = state.meta["selected_columns"]
            interactions = self._compute_interactions(df)
            # 只取 fit 时选中的列 — 测试集缺失的补 0
            available = [c for c in selected if c in interactions.columns]
            interaction_features = pd.DataFrame(0.0, index=df.index, columns=selected, dtype=np.float32)
            if available:
                interaction_features[available] = interactions[available].values
            logger.info("交互特征: %d 列 (训练时选中 %d, 测试集命中 %d)",
                        len(selected), len(selected), len(available))

        # ---- 拼接特征 ----
        # 移除 ID 列
        drop_cols = [c for c in self.id_cols if c in df.columns]
        if self.label_col in df.columns:
            drop_cols.append(self.label_col)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # 确保所有列为 float32
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df = df.astype(np.float32)

        # 拼接
        parts = [df]
        if tfidf_features is not None:
            parts.append(tfidf_features)
        if interaction_features is not None:
            parts.append(interaction_features)

        X_all = pd.concat(parts, axis=1)

        # ---- 6. 特征选择 ----
        state = self.states["feature_selector"]
        if state.fitted and state.meta.get("selected_columns"):
            selected = state.meta["selected_columns"]
            available = [c for c in selected if c in X_all.columns]
            missing_cols = set(selected) - set(available)
            if missing_cols:
                for mc in missing_cols:
                    X_all[mc] = 0.0
            X_all = X_all[selected]

        # ---- 7. SVD / PCA ----
        state = self.states["svd"]
        if state.fitted and state.transformer is not None:
            X_svd = state.transformer.transform(X_all)
            X_all = pd.DataFrame(X_svd, columns=state.column_names_after, index=X_all.index, dtype=np.float32)

        # ---- 最终排序 ----
        final_cols = [c for c in self.all_feature_names if c in X_all.columns]
        X_all = X_all[final_cols]

        logger.info("FeatureEngineer transform 完成 — 输出 shape=%s", X_all.shape)
        return X_all, y_result, event_ids

    # ================================================================
    # 内部辅助
    # ================================================================
    def _classify_columns(self, df: pd.DataFrame):
        """分类列：数值 / 低基数类别 / 高基数类别"""
        exclude = set(self.id_cols) | {self.label_col}
        all_cols = [c for c in df.columns if c not in exclude]

        cat_threshold = self.cfg.get("data", {}).get("categorical_threshold", 300)

        self.numeric_cols = []
        self.categorical_cols = []
        self.high_cardinality_cols = []
        self.low_cardinality_cols = []

        for col in all_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)
                n_unique = df[col].nunique(dropna=True)
                if n_unique > cat_threshold:
                    self.high_cardinality_cols.append(col)
                else:
                    self.low_cardinality_cols.append(col)

        self._column_classified = True
        logger.info(
            "列分类: numeric=%d, low_card=%d, high_card=%d",
            len(self.numeric_cols), len(self.low_cardinality_cols), len(self.high_cardinality_cols),
        )

    def _compute_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算数值列两两乘积"""
        num_cols = [c for c in df.columns if c not in set(self.id_cols) and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            return pd.DataFrame(index=df.index)

        interactions = {}
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                name = f"inter_{num_cols[i]}_{num_cols[j]}"
                interactions[name] = df[num_cols[i]] * df[num_cols[j]]

        return pd.DataFrame(interactions, index=df.index)

    def _apply_transforms(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """内部应用已有 transform 到当前数据（用于 fit 阶段选择特征）"""
        result = df.copy()

        # Scaler
        state = self.states["scaler"]
        if state.fitted and state.transformer is not None:
            fit_cols = list(state.column_names_before)
            available = [c for c in fit_cols if c in result.columns]
            if available:
                X_scaler = result[available].copy()
                for c in fit_cols:
                    if c not in X_scaler.columns:
                        X_scaler[c] = 0
                X_scaler = X_scaler[fit_cols]
                scaled = state.transformer.transform(X_scaler)
                scaled_df = pd.DataFrame(scaled, columns=fit_cols, index=result.index)
                for c in fit_cols:
                    result[c] = scaled_df[c]

        # Target Encoding
        state = self.states["target_encoder"]
        if state.fitted and state.transformer is not None:
            cols = [c for c in state.column_names_before if c in result.columns]
            if cols:
                result[cols] = state.transformer.transform(result[cols])

        # Low cardinality LE
        le_maps = state.meta.get("low_card_le_maps", {})
        for col, mapping in le_maps.items():
            if col in result.columns:
                result[col] = result[col].astype(str).fillna("UNKNOWN").map(mapping).fillna(-1).astype(int)

        # TF-IDF
        state = self.states["tfidf"]
        if state.fitted and state.transformer is not None and self.text_col in result.columns:
            text_data = result[self.text_col].fillna("").astype(str)
            tfidf_matrix = state.transformer.transform(text_data)
            if sp_sparse.issparse(tfidf_matrix):
                tfidf_matrix = tfidf_matrix.toarray()
            tfidf_df = pd.DataFrame(tfidf_matrix, columns=state.column_names_after, index=result.index, dtype=np.float32)
            result = result.drop(columns=[self.text_col], errors="ignore")
            result = pd.concat([result, tfidf_df], axis=1)

        # 移除 ID/label
        drop_cols = [c for c in self.id_cols if c in result.columns]
        if self.label_col in result.columns:
            drop_cols.append(self.label_col)
        result = result.drop(columns=drop_cols, errors="ignore")

        # 确保数值
        for c in result.columns:
            if not pd.api.types.is_numeric_dtype(result[c]):
                result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0)
        result = result.astype(np.float32)

        return result, None

    def _any_state_fitted(self) -> bool:
        return any(s.fitted for s in self.states.values()) or self._column_classified

    def _finalize_feature_names(self):
        """组合最终特征名列表"""
        names: List[str] = []

        # Scaler 列
        s = self.states["scaler"]
        if s.fitted:
            names.extend(s.column_names_after)

        # Target Encoding 列（替换了原低基数/高基数列的位置）
        s = self.states["target_encoder"]
        if s.fitted:
            names.extend(s.column_names_after)
        # 低基数 Label Encoded 列
        le_maps = s.meta.get("low_card_le_maps", {})
        names.extend(le_maps.keys())

        # TF-IDF
        s = self.states["tfidf"]
        if s.fitted:
            names.extend(s.column_names_after)

        # 交互特征
        s = self.states["interactions"]
        if s.fitted:
            names.extend(s.column_names_after)

        # 特征选择 — 替换上述所有
        s = self.states["feature_selector"]
        if s.fitted and s.meta.get("selected_columns"):
            names = list(s.column_names_after)

        # SVD — 最终替换
        s = self.states["svd"]
        if s.fitted:
            names = list(s.column_names_after)

        # 去重保序
        seen = set()
        deduped = []
        for n in names:
            if n not in seen:
                deduped.append(n)
                seen.add(n)

        self.all_feature_names = deduped

    # ================================================================
    # 保存 / 加载
    # ================================================================
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "feature_engineer.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FeatureEngineer 已保存: %s", path)

    @classmethod
    def load(cls, save_dir: str) -> "FeatureEngineer":
        path = os.path.join(save_dir, "feature_engineer.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"FeatureEngineer 文件不存在: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"加载的不是 FeatureEngineer 实例: {type(obj)}")
        logger.info("FeatureEngineer 已加载 (v%d), %d 最终特征", obj._version, len(obj.all_feature_names))
        return obj
