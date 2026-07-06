#!/usr/bin/env python3
"""
============================================================
SOC 网络安全威胁检测 — 训练主脚本 v2.0
============================================================
完整流程:
  1. 加载配置
  2. DataLoader fit/transform（含缺失值统计保存）
  3. 特征工程 fit/transform（含交互特征列锁定 + 特征选择）
  4. 不均衡处理（Hybrid / SMOTE-Tomek / SMOTE-ENN）
  5. Optuna 超参搜索 (可选)
  6. 多模型训练（RF, XGB, LGB, MLP）
  7. 五折交叉验证
  8. Voting Ensemble + Stacking
  9. 增量学习 partial_fit (可选)
 10. 可解释性分析
 11. 可视化报告输出
 12. 保存所有模型与产物
============================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import Config
from src.data_loader import DataLoader
from src.explainer import Explainer
from src.feature_engineer import FeatureEngineer
from src.imbalance_handler import ImbalanceHandler
from src.models import (
    ModelTrainer,
    HyperOptimizer,
    IncrementalTrainer,
    compute_metrics,
)
from src.visualizer import Visualizer

logger = logging.getLogger("train")


def setup_logging(cfg: Config):
    """配置日志"""
    log_level = getattr(logging, cfg.logging["level"], logging.INFO)
    log_format = cfg.logging["format"]

    os.makedirs(cfg.paths["log_dir"], exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(cfg.paths["log_dir"], "train.log"),
                encoding="utf-8",
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="SOC 威胁检测训练 v2.0")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--skip-cv", action="store_true", help="跳过交叉验证")
    parser.add_argument("--skip-explain", action="store_true", help="跳过可解释性")
    parser.add_argument("--no-ensemble", action="store_true", help="不构建集成")
    parser.add_argument("--no-stacking", action="store_true", help="不构建 Stacking")
    parser.add_argument("--incremental", action="store_true", help="启用增量学习")
    args = parser.parse_args()

    # ---- 1. 加载配置 ----
    cfg = Config(args.config)
    setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("SOC 网络安全威胁检测 v2.0 — 训练开始")
    logger.info("=" * 60)

    # ---- 2. DataLoader: fit + transform ----
    dl = DataLoader(cfg)
    train_df = dl.load_train()
    logger.info("训练集加载完成: %s", train_df.shape)

    # fit_transform 一步到位，内部保存所有统计信息
    X_train_full, y_train_full, X_val, y_val, X_full, y_full = dl.fulL_pipeline()

    if X_train_full is None or len(X_train_full) == 0:
        logger.error("训练数据为空，退出")
        return

    # ---- 3. 特征工程 ----
    fe = FeatureEngineer(cfg)

    # fit 在完整训练集上
    fe.fit(pd.concat([X_train_full, X_val]), pd.concat([y_train_full, y_val]))

    # transform 各部分
    X_train_feat, _, _ = fe.transform(X_train_full, y_train_full)
    X_val_feat, _, _ = fe.transform(X_val, y_val)

    feature_names = fe.all_feature_names
    logger.info("最终特征数: %d", len(feature_names))

    # ---- 4. 不均衡处理 ----
    imh = ImbalanceHandler(cfg)

    # 记录重采样前分布
    pre_dist = y_train_full.values.copy() if hasattr(y_train_full, "values") else np.asarray(y_train_full).copy()

    # 对训练集执行重采样
    X_np = X_train_feat.values if hasattr(X_train_feat, "values") else np.asarray(X_train_feat)
    y_np = y_train_full.values if hasattr(y_train_full, "values") else np.asarray(y_train_full)

    if cfg.imbalance.method != "class_weight":
        X_train_res_np, y_train_res_np = imh.resample(X_np, y_np)
        X_train_res = pd.DataFrame(X_train_res_np, columns=feature_names).astype(np.float32)
        y_train_res = pd.Series(y_train_res_np)
    else:
        X_train_res, y_train_res = X_train_feat, y_train_full

    logger.info("训练集最终形状 (resample后): %s", X_train_res.shape)

    # ---- 5. 多模型构建 ----
    trainer = ModelTrainer(cfg)
    models = trainer.build_models()

    # 转 numpy
    X_train_np = X_train_res.values if hasattr(X_train_res, "values") else np.asarray(X_train_res)
    y_train_np = y_train_res.values if hasattr(y_train_res, "values") else np.asarray(y_train_res)
    X_val_np = X_val_feat.values if hasattr(X_val_feat, "values") else np.asarray(X_val_feat)
    y_val_np = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)

    # ---- 6. 超参搜索 (可选) ----
    trainer.hyperopt(X_train_np, y_train_np)

    # ---- 7. 训练 + CV ----
    for name, model in models.items():
        metrics = trainer.train_single(name, model, X_train_np, y_train_np, X_val_np, y_val_np)
        logger.info(
            "%s → Acc: %.4f  F1: %.4f  F1(mal): %.4f  AUC: %.4f",
            name,
            metrics.get("accuracy", 0),
            metrics.get("f1_macro", 0),
            metrics.get("f1_malicious", 0),
            metrics.get("auc_ovr", 0),
        )

        if not args.skip_cv:
            trainer.cross_validate(name, model, X_train_np, y_train_np)

    # ---- 8. 集成 ----
    best_name, best_model = trainer.select_best()

    # Voting Ensemble
    if not args.no_ensemble and len(trainer.trained_models) >= 2:
        trainer.build_ensemble()
        ensemble_metrics = trainer.train_ensemble(X_train_np, y_train_np, X_val_np, y_val_np)
        logger.info(
            "Ensemble → Acc: %.4f  F1: %.4f  F1(mal): %.4f  AUC: %.4f",
            ensemble_metrics.get("accuracy", 0),
            ensemble_metrics.get("f1_macro", 0),
            ensemble_metrics.get("f1_malicious", 0),
            ensemble_metrics.get("auc_ovr", 0),
        )

    # Stacking
    if not args.no_stacking and len(trainer.trained_models) >= 2:
        trainer.build_stacking()
        stacking_metrics = trainer.train_stacking(X_train_np, y_train_np, X_val_np, y_val_np)

    # ---- 9. 增量学习 (可选) ----
    inc_model = None
    if args.incremental or cfg.get("incremental", {}).get("enabled", False):
        logger.info("启动增量学习...")
        inc = IncrementalTrainer(cfg, n_classes=len(np.unique(y_train_np)), n_features=X_train_np.shape[1])
        inc.build()
        inc.train(X_train_np, y_train_np)
        y_pred_inc = inc.predict(X_val_np)
        y_prob_inc = inc.predict_proba(X_val_np)
        inc_metrics = compute_metrics(y_val_np, y_pred_inc, y_prob_inc)
        trainer.metrics["incremental"] = inc_metrics
        logger.info("增量学习 → F1: %.4f  F1(mal): %.4f",
                    inc_metrics.get("f1_macro", 0), inc_metrics.get("f1_malicious", 0))

    # ---- 10. 测试集评估（用验证集近似）----
    y_pred_test, y_prob_test = trainer.predict(
        X_val_np,
        use_ensemble=not args.no_ensemble,
        use_stacking=not args.no_stacking,
    )
    test_metrics = compute_metrics(y_val_np, y_pred_test, y_prob_test)
    logger.info(
        "Test Set → Acc: %.4f  F1: %.4f  F1(malicious): %.4f  AUC: %.4f",
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
        test_metrics["f1_malicious"],
        test_metrics["auc_ovr"],
    )
    trainer.metrics["test"] = test_metrics

    # ---- 11. 可解释性 ----
    importance_df = None
    if not args.skip_explain:
        explainer = Explainer(cfg)
        model_to_explain = (
            trainer.stacking if (not args.no_stacking and trainer.stacking is not None)
            else trainer.ensemble if (not args.no_ensemble and trainer.ensemble is not None)
            else best_model
        )
        explain_results = explainer.explain(model_to_explain, X_train_np, X_val_np, feature_names)
        explainer.save_results(explain_results, cfg.paths["viz_dir"])

        # 提取重要性
        if "shap_tree" in explain_results and explain_results["shap_tree"] is not None:
            importance_df = explain_results["shap_tree"]
        elif "permutation" in explain_results and explain_results["permutation"] is not None:
            importance_df = explain_results["permutation"]

    # ---- 12. 可视化 ----
    viz = Visualizer(cfg)
    viz.generate_report(
        metrics_dict=trainer.metrics,
        y_true=y_val_np,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
        importance_df=importance_df,
        pre_resample_dist=pre_dist,
        post_resample_dist=y_train_np,
    )

    # ---- 13. 保存 ----
    os.makedirs(cfg.paths["model_dir"], exist_ok=True)
    trainer.save(cfg.paths["model_dir"])
    fe.save(cfg.paths["model_dir"])
    dl.save(cfg.paths["model_dir"])

    # 保存 metrics
    metrics_serializable = {}
    for name, m in trainer.metrics.items():
        sm = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                sm[k] = v.tolist()
            elif isinstance(v, (np.integer,)):
                sm[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sm[k] = float(v)
            elif isinstance(v, (np.float32, np.float64)):
                sm[k] = float(v)
            else:
                sm[k] = v
        metrics_serializable[name] = sm

    out_dir = cfg.paths.get("output_dir") or cfg.paths.get("model_dir", "models")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("训练完成！最佳模型: %s | 最终特征数: %d",
                best_name or "ensemble", len(feature_names))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
