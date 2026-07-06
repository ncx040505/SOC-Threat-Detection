"""
模型模块 — 训练 / 评估 / 集成 / Stacking / 超参搜索 / 增量学习
✅ v2.0 新增:
  - Stacking 二级集成
  - Optuna 超参搜索
  - 增量学习 partial_fit
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.config_loader import Config

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ================================================================
# 指标计算
# ================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    """计算多分类指标"""
    if labels is None:
        labels = [0, 1, 2]

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # per-class
    for k, v in classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0).items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(v, dict):
            metrics[f"precision_{k}"] = v.get("precision", 0)
            metrics[f"recall_{k}"] = v.get("recall", 0)
            metrics[f"f1_{k}"] = v.get("f1-score", 0)

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == len(np.unique(y_true)):
        try:
            metrics["auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            metrics["auc_ovr"] = 0.0

    # 恶意类 F1
    ml_idx = 1 if 1 in labels else 0
    ml_label = "malicious" if "malicious" in labels else labels[ml_idx] if len(labels) > ml_idx else "1"
    metrics["f1_malicious"] = metrics.get(f"f1_{ml_label}", 0)

    return metrics


# ================================================================
# 超参搜索 (Optuna)
# ================================================================
class HyperOptimizer:
    """Optuna 超参搜索"""

    def __init__(self, config: Config, X: np.ndarray, y: np.ndarray):
        self.cfg = config
        self.X = X
        self.y = y
        self.cv_folds = config.get("hyperopt", {}).get("cv_folds", 3)
        self.n_trials = config.get("hyperopt", {}).get("n_trials", 50)
        self.timeout = config.get("hyperopt", {}).get("timeout", 3600)
        self.metric = config.get("hyperopt", {}).get("metric", "f1_macro")

    def optimize(self, model_type: str = "xgboost") -> Tuple[Dict, float]:
        """运行 Optuna 搜索"""
        if not _HAS_OPTUNA:
            logger.warning("optuna 未安装，跳过超参搜索")
            return {}, 0.0

        logger.info("Optuna 超参搜索开始: %s, %d trials", model_type, self.n_trials)

        if model_type == "xgboost" and _HAS_XGB:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._xgboost_objective(trial),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,
            )
        elif model_type == "lightgbm" and _HAS_LGB:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._lgb_objective(trial),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,
            )
        elif model_type == "random_forest":
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._rf_objective(trial),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,
            )
        else:
            logger.warning("不支持的模型类型: %s", model_type)
            return {}, 0.0

        logger.info("Optuna 完成 — 最佳 score=%.4f, params=%s", study.best_value, study.best_params)
        return study.best_params, study.best_value

    def _cv_score(self, model) -> float:
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, X_va = self.X[train_idx], self.X[val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_va)
            scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
        return np.mean(scores)

    def _xgboost_objective(self, trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        return self._cv_score(model)

    def _lgb_objective(self, trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        return self._cv_score(model)

    def _rf_objective(self, trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        return self._cv_score(model)


# ================================================================
# 增量学习
# ================================================================
class IncrementalTrainer:
    """支持 partial_fit 的增量学习"""

    def __init__(self, config: Config, n_classes: int, n_features: int):
        self.cfg = config
        self.n_classes = n_classes
        self.n_features = n_features
        self.batch_size = config.get("incremental", {}).get("batch_size", 1000)
        self.epochs_per_batch = config.get("incremental", {}).get("epochs_per_batch", 5)
        self.model: Optional[BaseEstimator] = None

    def build(self) -> BaseEstimator:
        model_type = self.cfg.get("incremental", {}).get("model", "mlp")
        if model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=1,
                warm_start=True,
                random_state=42,
            )
        elif model_type == "sgd":
            model = SGDClassifier(
                loss="log_loss",
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
            )
        else:
            raise ValueError(f"不支持的增量学习模型: {model_type}")
        self.model = model
        return model

    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.build()

        n_samples = len(X)
        classes = np.unique(y)

        for epoch in range(self.epochs_per_batch):
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                self.model.partial_fit(X_batch, y_batch, classes=classes)

        logger.info("增量学习完成: %d epochs × %d samples", self.epochs_per_batch, n_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        d = self.model.decision_function(X)
        from scipy.special import softmax
        return softmax(d, axis=1)


# ================================================================
# 模型训练器
# ================================================================
class ModelTrainer:
    """多模型训练 + 集成 + Stacking"""

    def __init__(self, config: Config):
        self.cfg = config
        self.models: Dict[str, Any] = {}
        self.trained_models: Dict[str, Any] = {}
        self.ensemble: Optional[Any] = None
        self.stacking: Optional[Any] = None
        self.metrics: Dict[str, Dict] = {}
        self.cv_results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_score: float = -1.0

    # ----------------------------------------------------------------
    # 构建模型
    # ----------------------------------------------------------------
    def build_models(self) -> Dict[str, Any]:
        """根据配置构建所有启用的模型"""
        enabled = self.cfg.get("models", {}).get("enabled", ["random_forest", "xgboost", "lightgbm", "mlp"])
        cfg_models = self.cfg.get("models", {})

        for name in enabled:
            if name == "random_forest":
                params = cfg_models.get("random_forest", {})
                self.models[name] = RandomForestClassifier(**{k: v for k, v in dict(params).items() if k != "enabled"})
            elif name == "xgboost" and _HAS_XGB:
                params = cfg_models.get("xgboost", {})
                self.models[name] = xgb.XGBClassifier(**{k: v for k, v in dict(params).items() if k != "enabled"})
            elif name == "lightgbm" and _HAS_LGB:
                params = cfg_models.get("lightgbm", {})
                lgb_params = {k: v for k, v in dict(params).items() if k != "enabled"}
                if "verbose" not in lgb_params:
                    lgb_params["verbose"] = -1
                self.models[name] = lgb.LGBMClassifier(**lgb_params)
            elif name == "mlp":
                params = cfg_models.get("mlp", {})
                mlp_params = {
                    "hidden_layer_sizes": tuple(params.get("hidden_layers", [256, 128, 64])),
                    "activation": params.get("activation", "relu"),
                    "batch_size": params.get("batch_size", 256),
                    "learning_rate_init": params.get("learning_rate", 0.001),
                    "max_iter": params.get("epochs", 200),
                    "early_stopping": True,
                    "random_state": params.get("random_state", 42),
                }
                self.models[name] = MLPClassifier(**mlp_params)

        logger.info("已构建 %d 个模型: %s", len(self.models), list(self.models.keys()))
        return self.models

    # ----------------------------------------------------------------
    # 超参搜索
    # ----------------------------------------------------------------
    def hyperopt(self, X: np.ndarray, y: np.ndarray):
        """对每个模型执行超参搜索，替换默认参数"""
        opt_cfg = self.cfg.get("hyperopt", {})
        if not opt_cfg.get("enabled", False):
            return

        optimizer = HyperOptimizer(self.cfg, X, y)
        for name in list(self.models.keys()):
            logger.info("超参搜索: %s", name)
            try:
                best_params, best_score = optimizer.optimize(model_type=name)
                if best_params:
                    self.models[name].set_params(**best_params)
                    logger.info("%s 最佳参数: %s (score=%.4f)", name, best_params, best_score)
            except Exception as e:
                logger.warning("%s 超参搜索失败: %s", name, e)

    # ----------------------------------------------------------------
    # 训练单模型
    # ----------------------------------------------------------------
    def train_single(
        self,
        name: str,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """训练单个模型并评估"""
        logger.info("训练 %s ...", name)
        t0 = time.time()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics["train_time"] = time.time() - t0

        self.trained_models[name] = model
        self.metrics[name] = metrics

        primary = self.cfg.get("training", {}).get("primary_metric", "f1_macro")
        score = metrics.get(primary, 0)
        if score > self.best_score:
            self.best_score = score
            self.best_model_name = name

        return metrics

    # ----------------------------------------------------------------
    # 交叉验证
    # ----------------------------------------------------------------
    def cross_validate(self, name: str, model, X: np.ndarray, y: np.ndarray) -> Dict:
        """分层 K 折交叉验证"""
        n_folds = self.cfg.get("training", {}).get("cv_folds", 5)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scoring = ["accuracy", "f1_macro", "f1_weighted"]

        # 编码标签为连续整数
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        t0 = time.time()
        scores = cross_validate(
            model, X, y_enc, cv=skf, scoring=scoring,
            n_jobs=-1, return_train_score=False, verbose=0,
        )
        elapsed = time.time() - t0

        result = {
            "cv_folds": n_folds,
            "cv_time": elapsed,
        }
        for key in scoring:
            result[f"mean_{key}"] = scores[f"test_{key}"].mean()
            result[f"std_{key}"] = scores[f"test_{key}"].std()

        self.cv_results[name] = result
        logger.info(
            "CV %s: F1=%.4f±%.4f, Acc=%.4f±%.4f",
            name,
            result["mean_f1_macro"],
            result["std_f1_macro"],
            result["mean_accuracy"],
            result["std_accuracy"],
        )
        return result

    # ----------------------------------------------------------------
    # 选择最佳模型
    # ----------------------------------------------------------------
    def select_best(self) -> Tuple[Optional[str], Optional[Any]]:
        if not self.trained_models:
            return None, None
        primary = self.cfg.get("training", {}).get("primary_metric", "f1_macro")
        best = max(
            self.trained_models.keys(),
            key=lambda n: self.metrics[n].get(primary, 0),
        )
        self.best_model_name = best
        logger.info("最佳单模型: %s (score=%.4f)", best, self.metrics[best].get(primary, 0))
        return best, self.trained_models[best]

    # ----------------------------------------------------------------
    # Voting Ensemble
    # ----------------------------------------------------------------
    def build_ensemble(self):
        """构建软投票集成"""
        if len(self.trained_models) < 2:
            logger.warning("模型数不足 2，无法构建集成")
            return None

        estimators = [(name, model) for name, model in self.trained_models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=None,  # 所有模型等权
        )
        # VotingClassifier 需要拟合（实际只拟合每个子模型的 weights）
        self.ensemble.estimators_ = [m for _, m in estimators]
        self.ensemble.le_ = LabelEncoder().fit([0, 1, 2])
        logger.info("构建 Soft Voting 集成: %d 个模型", len(estimators))
        return self.ensemble

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """评估 Voting Ensemble"""
        if self.ensemble is None:
            self.build_ensemble()
        if self.ensemble is None:
            return {}

        y_pred = self.ensemble.predict(X_val)
        y_prob = self.ensemble.predict_proba(X_val) if hasattr(self.ensemble, "predict_proba") else None
        metrics = compute_metrics(y_val, y_pred, y_prob)
        self.metrics["voting_ensemble"] = metrics
        return metrics

    # ----------------------------------------------------------------
    # Stacking
    # ----------------------------------------------------------------
    def build_stacking(self, n_classes: int = 3):
        """构建 Stacking 集成"""
        if len(self.trained_models) < 2:
            logger.warning("模型数不足 2，无法构建 Stacking")
            return None

        scfg = self.cfg.get("stacking", {})
        if not scfg.get("enabled", True):
            return None

        final_type = scfg.get("final_estimator", "logistic")
        if final_type == "logistic":
            final = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        elif final_type == "lightgbm" and _HAS_LGB:
            final = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        elif final_type == "xgboost" and _HAS_XGB:
            final = xgb.XGBClassifier(n_estimators=100, tree_method="hist", random_state=42)
        else:
            final = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)

        estimators = [(name, model) for name, model in self.trained_models.items()]
        stack_method = scfg.get("stack_method", "predict_proba")
        passthrough = scfg.get("passthrough", True)
        cv = scfg.get("cv", 5)

        self.stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=final,
            cv=cv,
            stack_method=stack_method,
            passthrough=passthrough,
            n_jobs=-1,
        )
        logger.info("构建 Stacking 集成: %d 基模型, final=%s, passthrough=%s", len(estimators), final_type, passthrough)
        return self.stacking

    def train_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """训练 & 评估 Stacking"""
        if self.stacking is None:
            self.build_stacking()
        if self.stacking is None:
            return {}

        logger.info("训练 Stacking ...")
        t0 = time.time()
        self.stacking.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = self.stacking.predict(X_val)
        y_prob = self.stacking.predict_proba(X_val) if hasattr(self.stacking, "predict_proba") else None
        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics["train_time"] = elapsed
        self.metrics["stacking"] = metrics
        logger.info("Stacking 完成 (%.1fs), F1=%.4f", elapsed, metrics.get("f1_macro", 0))
        return metrics

    # ----------------------------------------------------------------
    # 预测
    # ----------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        use_ensemble: bool = True,
        use_stacking: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        用最佳模型或集成预测。
        返回 (y_pred, y_prob)
        """
        # 优先级: Stacking > Voting Ensemble > 最佳单模型
        if use_stacking and self.stacking is not None and hasattr(self.stacking, "predict"):
            y_pred = self.stacking.predict(X)
            y_prob = self.stacking.predict_proba(X) if hasattr(self.stacking, "predict_proba") else None
        elif use_ensemble and self.ensemble is not None:
            y_pred = self.ensemble.predict(X)
            y_prob = self.ensemble.predict_proba(X) if hasattr(self.ensemble, "predict_proba") else None
        elif self.best_model_name and self.best_model_name in self.trained_models:
            model = self.trained_models[self.best_model_name]
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        else:
            model = list(self.trained_models.values())[0]
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        if y_prob is None:
            y_prob = np.zeros((len(X), 3))

        return y_pred, y_prob

    # ----------------------------------------------------------------
    # 保存/加载
    # ----------------------------------------------------------------
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.trained_models.items():
            path = os.path.join(save_dir, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

        if self.ensemble is not None:
            path = os.path.join(save_dir, "voting_ensemble.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.ensemble, f)

        if self.stacking is not None:
            path = os.path.join(save_dir, "stacking.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.stacking, f)

        # 保存元数据
        meta = {
            "best_model_name": self.best_model_name,
            "best_score": self.best_score,
            "model_names": list(self.trained_models.keys()),
            "has_ensemble": self.ensemble is not None,
            "has_stacking": self.stacking is not None,
        }
        import json
        with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("模型已保存: %s (%d 个)", save_dir, len(self.trained_models))

    def load(self, save_dir: str):
        import json
        meta_path = os.path.join(save_dir, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.best_model_name = meta.get("best_model_name")
            self.best_score = meta.get("best_score", -1)

        for mname in meta.get("model_names", []):
            path = os.path.join(save_dir, f"{mname}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.trained_models[mname] = pickle.load(f)

        ens_path = os.path.join(save_dir, "voting_ensemble.pkl")
        if os.path.exists(ens_path):
            with open(ens_path, "rb") as f:
                self.ensemble = pickle.load(f)

        stk_path = os.path.join(save_dir, "stacking.pkl")
        if os.path.exists(stk_path):
            with open(stk_path, "rb") as f:
                self.stacking = pickle.load(f)

        logger.info("模型已加载: %s (%d 个)", save_dir, len(self.trained_models))
