"""
可解释性模块
✅ v2.0: 改善稳定性 — SHAP 采样降级、异常隔离、top_features 可配置
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config_loader import Config

logger = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.warning("shap 未安装，可解释性功能受限")

try:
    from sklearn.inspection import permutation_importance
except ImportError:
    permutation_importance = None


class Explainer:
    """模型可解释性"""

    def __init__(self, config: Config):
        self.cfg = config
        self.top_n = config.get("explainability", {}).get("top_features", 30)
        self.shap_sample = config.get("explainability", {}).get("shap_sample_size", 2000)

    def explain(
        self,
        model,
        X_train: np.ndarray,
        X_val: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """执行所有可解释性方法"""
        results = {}
        methods = self.cfg.get("explainability", {}).get("methods", ["shap_tree", "permutation"])

        if "shap_tree" in methods:
            results["shap_tree"] = self._shap_tree(model, X_train, X_val, feature_names)
        if "shap_kernel" in methods:
            results["shap_kernel"] = self._shap_kernel(model, X_train, X_val, feature_names)
        if "permutation" in methods:
            results["permutation"] = self._permutation_importance(model, X_val, feature_names)

        return results

    def _shap_tree(
        self,
        model,
        X_train: np.ndarray,
        X_val: np.ndarray,
        feature_names: List[str],
    ) -> Optional[pd.DataFrame]:
        """Tree SHAP — 对于树模型最快"""
        if not _HAS_SHAP:
            logger.warning("SHAP 不可用")
            return None

        try:
            # 降采样背景数据
            n_bg = min(self.shap_sample, len(X_train))
            background = X_train[np.random.choice(len(X_train), n_bg, replace=False)]

            explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_val[:self.shap_sample])

            # 多分类 SHAP 返回 list of arrays
            if isinstance(shap_values, list):
                # 取 malicious 类 (index 1)
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
                if shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]

            importance = np.abs(shap_vals).mean(axis=0)
            df = pd.DataFrame({
                "feature": feature_names[:len(importance)],
                "shap_importance": importance,
            }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

            logger.info("Tree SHAP 完成, top-5: %s", df.head(5)["feature"].tolist())
            return df
        except Exception as e:
            logger.warning("Tree SHAP 失败: %s，回退到 Kernel SHAP", e)
            return self._shap_kernel(model, X_train, X_val, feature_names)

    def _shap_kernel(
        self,
        model,
        X_train: np.ndarray,
        X_val: np.ndarray,
        feature_names: List[str],
    ) -> Optional[pd.DataFrame]:
        """Kernel SHAP — 适用于任意模型"""
        if not _HAS_SHAP:
            return None

        try:
            n_bg = min(100, len(X_train))
            n_test = min(500, len(X_val))
            background = X_train[np.random.choice(len(X_train), n_bg, replace=False)]
            test_data = X_val[:n_test]

            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x) if hasattr(model, "predict_proba") else model.decision_function(x),
                background,
            )
            shap_values = explainer.shap_values(test_data, nsamples=100)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            importance = np.abs(shap_vals).mean(axis=0)
            df = pd.DataFrame({
                "feature": feature_names[:len(importance)],
                "kernel_shap_importance": importance,
            }).sort_values("kernel_shap_importance", ascending=False).reset_index(drop=True)

            logger.info("Kernel SHAP 完成 (nsamples=100)")
            return df
        except Exception as e:
            logger.warning("Kernel SHAP 失败: %s", e)
            return None

    def _permutation_importance(
        self,
        model,
        X_val: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Permutation Importance"""
        try:
            n_use = min(20000, len(X_val))
            y_pred = model.predict(X_val[:n_use])
            y_true = y_pred  # 用预测值做近似

            importances = permutation_importance(
                model,
                X_val[:n_use],
                y_true,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1,
                scoring="f1_macro",
            )

            df = pd.DataFrame({
                "feature": feature_names[:len(importances.importances_mean)],
                "permutation_importance": importances.importances_mean,
                "permutation_std": importances.importances_std,
            }).sort_values("permutation_importance", ascending=False).reset_index(drop=True)

            logger.info("Permutation 完成, top-5: %s", df.head(5)["feature"].tolist())
            return df
        except Exception as e:
            logger.warning("Permutation 失败: %s", e)
            return None

    def save_results(self, results: Dict[str, Any], output_dir: str):
        """保存可解释性结果"""
        os.makedirs(output_dir, exist_ok=True)
        for name, df in results.items():
            if df is not None and isinstance(df, pd.DataFrame):
                # 保存 CSV
                csv_path = os.path.join(output_dir, f"explain_{name}.csv")
                df.to_csv(csv_path, index=False)
                logger.info("可解释性结果已保存: %s", csv_path)

                # 保存 JSON（top-N）
                json_path = os.path.join(output_dir, f"explain_{name}_top{self.top_n}.json")
                top = df.head(self.top_n)
                top.to_json(json_path, orient="records", indent=2)
