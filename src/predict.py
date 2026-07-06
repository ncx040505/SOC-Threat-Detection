#!/usr/bin/env python3
"""
============================================================
SOC 网络安全威胁检测 — 预测脚本 v2.0
============================================================
读取测试集 → 加载 DataLoader 状态 + 特征工程 + 模型 → 预测 → 输出 res.csv

v2.0 改进:
  - 严格使用训练时保存的 DataLoader 统计信息
  - 交互特征列名锁定
  - 支持 Stacking / Voting / 单模型切换
  - 可输出概率
============================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import Config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.models import ModelTrainer

logger = logging.getLogger("predict")


def main():
    parser = argparse.ArgumentParser(description="SOC 威胁检测预测 v2.0")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--test-file", type=str, default=None, help="测试集路径（覆盖配置）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（覆盖配置）")
    parser.add_argument("--model-dir", type=str, default=None, help="模型目录（覆盖配置）")
    parser.add_argument("--use-best", action="store_true", help="使用最佳单模型")
    parser.add_argument("--use-stacking", action="store_true", help="使用 Stacking 集成")
    parser.add_argument("--probabilities", action="store_true", help="同时输出各类别概率")
    args = parser.parse_args()

    cfg = Config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    model_dir = args.model_dir or cfg.paths["model_dir"]

    # ---- 1. 加载 DataLoader 状态 ----
    logger.info("加载 DataLoader 状态: %s", model_dir)
    dl = DataLoader.load(cfg, model_dir)

    # 读取测试数据
    if args.test_file:
        test_path = args.test_file
    else:
        test_path = os.path.join(cfg.paths["raw_dir"], cfg.get("data", {}).get("test_file", "test.csv"))

    logger.info("读取测试集: %s", test_path)
    df_test = dl._read_file(test_path)

    # ---- 2. 用 DataLoader 状态 transform ----
    X_test, _, event_ids = dl.transform(df_test, has_label=False)

    # ---- 3. 加载特征工程 ----
    logger.info("加载特征工程: %s", model_dir)
    fe = FeatureEngineer.load(model_dir)
    X_test_feat, _, event_ids = fe.transform(X_test, event_ids=event_ids)

    X_np = X_test_feat.values if hasattr(X_test_feat, "values") else np.asarray(X_test_feat)
    logger.info("测试特征 shape: %s", X_np.shape)

    # ---- 4. 加载模型 ----
    logger.info("加载模型: %s", model_dir)
    trainer = ModelTrainer(cfg)
    trainer.load(model_dir)

    # ---- 5. 预测 ----
    use_ensemble = not args.use_best
    use_stacking = args.use_stacking
    y_pred, y_prob = trainer.predict(X_np, use_ensemble=use_ensemble, use_stacking=use_stacking)

    # 解码标签
    labels = [dl.label_decoder.get(int(p), "benign") for p in y_pred]

    # ---- 6. 输出 res.csv ----
    output_path = args.output or os.path.join(cfg.paths["output_dir"], cfg.prediction["output_file"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    res_df = pd.DataFrame({
        "event_id": event_ids.values if hasattr(event_ids, "values") else event_ids,
        "pred_label": labels,
    })

    # 概率输出
    if args.probabilities or cfg.prediction.get("output_probabilities", False):
        for i, cls_name in enumerate(["benign", "malicious", "suspicious"]):
            res_df[f"prob_{cls_name}"] = y_prob[:, i]

    res_df.to_csv(output_path, index=False)
    logger.info("预测结果已保存: %s (%d rows)", output_path, len(res_df))

    # 分布统计
    dist = res_df["pred_label"].value_counts()
    logger.info("预测分布:\n%s", dist.to_string())

    if args.probabilities or cfg.prediction.get("output_probabilities", False):
        logger.info("恶意事件 TOP-10:")
        top_malicious = res_df.nlargest(10, "prob_malicious")[
            ["event_id", "pred_label", "prob_malicious", "prob_suspicious"]
        ]
        logger.info("\n%s", top_malicious.to_string())


if __name__ == "__main__":
    main()
