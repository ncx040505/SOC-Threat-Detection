"""
不均衡处理模块
✅ v2.0: SMOTE+Tomek / SMOTE+ENN 支持
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.utils import class_weight as cw_util

from src.config_loader import Config

logger = logging.getLogger(__name__)

try:
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.over_sampling import (
        ADASYN,
        SMOTE,
        BorderlineSMOTE,
        RandomOverSampler,
    )
    from imblearn.under_sampling import RandomUnderSampler

    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False
    logger.warning("imbalanced-learn 未安装，重采样功能不可用")


class ImbalanceHandler:
    """不均衡处理"""

    def __init__(self, config: Config):
        self.cfg = config
        self.class_weights: Optional[Dict[int, float]] = None

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """计算 class_weight"""
        classes = np.unique(y)
        weights = cw_util.compute_class_weight("balanced", classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        logger.info("Class weights: %s", {int(k): float(v) for k, v in self.class_weights.items()})
        return self.class_weights

    def resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据配置执行重采样。
        """
        method = self.cfg.get("imbalance", {}).get("method", "hybrid")
        seed = self.cfg.get("imbalance", {}).get("random_state", 42)

        if method == "class_weight":
            self.compute_class_weights(y)
            return X, y

        if not _HAS_IMBLEARN:
            logger.warning("imblearn 不可用，回退到 class_weight 模式")
            self.compute_class_weights(y)
            return X, y

        logger.info("重采样方法: %s, 输入分布: %s", method, dict(zip(*np.unique(y, return_counts=True))))

        try:
            if method == "smote":
                sampler = SMOTE(random_state=seed, n_jobs=-1)
            elif method == "adasyn":
                sampler = ADASYN(random_state=seed, n_jobs=-1)
            elif method == "borderline_smote":
                sampler = BorderlineSMOTE(random_state=seed, n_jobs=-1)
            elif method == "random_over":
                sampler = RandomOverSampler(random_state=seed)
            elif method == "random_under":
                sampler = RandomUnderSampler(random_state=seed)
            elif method == "hybrid":
                return self._hybrid_resample(X, y, seed)
            elif method == "smote_tomek":
                sampler = SMOTETomek(random_state=seed, n_jobs=-1)
            elif method == "smote_enn":
                sampler = SMOTEENN(random_state=seed, n_jobs=-1)
            else:
                logger.warning("未知重采样方法: %s，返回原始数据", method)
                return X, y

            if method not in ("hybrid",):
                X_res, y_res = sampler.fit_resample(X, y)
                logger.info("重采样完成: %s → %s", X.shape, X_res.shape)
                return X_res, y_res

        except Exception as e:
            logger.error("重采样失败: %s，返回原始数据", e)
            return X, y

        return X, y

    def _hybrid_resample(self, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid 混合采样"""
        hcfg = self.cfg.get("imbalance", {}).get("hybrid", {})

        under_strategy = hcfg.get("under_sampling_strategy", {0: 50000})
        over_strategy = hcfg.get("over_sampling_strategy", {1: 30000, 2: 30000})

        # Step 1: 欠采样多数类
        under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
        try:
            X_u, y_u = under.fit_resample(X, y)
            logger.info("欠采样完成: %s → %s, 分布: %s", X.shape, X_u.shape, dict(zip(*np.unique(y_u, return_counts=True))))
        except Exception as e:
            logger.warning("欠采样失败: %s, 跳过", e)
            X_u, y_u = X, y

        # Step 2: SMOTE 过采样少数类
        over = SMOTE(sampling_strategy=over_strategy, random_state=seed, n_jobs=-1)
        try:
            X_res, y_res = over.fit_resample(X_u, y_u)
            logger.info("过采样完成: %s → %s, 分布: %s", X_u.shape, X_res.shape, dict(zip(*np.unique(y_res, return_counts=True))))
        except Exception as e:
            logger.warning("过采样失败: %s, 跳过", e)
            X_res, y_res = X_u, y_u

        return X_res, y_res
