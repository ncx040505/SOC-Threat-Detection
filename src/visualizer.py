"""
可视化模块
✅ v2.0: 增加 ROC 曲线、Stacking 对比、时间分析
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)

from src.config_loader import Config

logger = logging.getLogger(__name__)


class Visualizer:
    """可视化报告生成"""

    def __init__(self, config: Config):
        self.cfg = config
        self.output_dir = config.paths["viz_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # 中文字体设置
        try:
            plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        sns.set_style("whitegrid")

    def generate_report(
        self,
        metrics_dict: Dict[str, Dict],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        importance_df: Optional[pd.DataFrame] = None,
        pre_resample_dist: Optional[np.ndarray] = None,
        post_resample_dist: Optional[np.ndarray] = None,
    ):
        """生成完整可视化报告"""
        logger.info("开始生成可视化报告...")

        # 1. 标签分布
        self._plot_label_distribution(pre_resample_dist, post_resample_dist)

        # 2. 混淆矩阵
        self._plot_confusion_matrix(y_true, y_pred)

        # 3. 模型对比
        self._plot_model_comparison(metrics_dict)

        # 4. 特征重要性
        if importance_df is not None:
            self._plot_feature_importance(importance_df)

        # 5. ROC 曲线
        if y_prob is not None:
            self._plot_roc_curves(y_true, y_prob)

        logger.info("可视化报告生成完成 → %s", self.output_dir)

    def _plot_label_distribution(
        self,
        pre_dist: Optional[np.ndarray],
        post_dist: Optional[np.ndarray],
    ):
        """标签分布对比"""
        if pre_dist is None and post_dist is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        label_names = ["benign", "malicious", "suspicious"]

        if pre_dist is not None:
            unique_pre, counts_pre = np.unique(pre_dist, return_counts=True)
            pre_counts = {label_names[i] if i < len(label_names) else str(i): c for i, c in zip(unique_pre, counts_pre)}
            ax = axes[0]
            bars = ax.bar(pre_counts.keys(), pre_counts.values(), color=["#2ecc71", "#e74c3c", "#f39c12"])
            ax.set_title("Before Resampling")
            ax.set_ylabel("Count")
            for bar, val in zip(bars, pre_counts.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + val * 0.01,
                        f"{val:,}", ha="center", fontsize=9)

        if post_dist is not None:
            unique_post, counts_post = np.unique(post_dist, return_counts=True)
            post_counts = {label_names[i] if i < len(label_names) else str(i): c for i, c in zip(unique_post, counts_post)}
            ax = axes[1]
            bars = ax.bar(post_counts.keys(), post_counts.values(), color=["#2ecc71", "#e74c3c", "#f39c12"])
            ax.set_title("After Resampling")
            ax.set_ylabel("Count")
            for bar, val in zip(bars, post_counts.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + val * 0.01,
                        f"{val:,}", ha="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "label_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("标签分布图已保存: %s", path)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["benign", "malicious", "suspicious"],
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix (Test Set)")

        path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("混淆矩阵已保存: %s", path)

    def _plot_model_comparison(self, metrics_dict: Dict[str, Dict]):
        """模型对比（F1 / Accuracy / AUC）"""
        model_names = list(metrics_dict.keys())
        if not model_names:
            return

        metrics_to_plot = ["accuracy", "f1_macro", "f1_malicious", "auc_ovr"]
        avaliable_metrics = [m for m in metrics_to_plot if any(m in v for v in metrics_dict.values())]

        fig, axes = plt.subplots(1, len(avaliable_metrics), figsize=(5 * len(avaliable_metrics), 5))
        if len(avaliable_metrics) == 1:
            axes = [axes]

        colors = sns.color_palette("Set2", len(model_names))

        for ax, metric in zip(axes, avaliable_metrics):
            values = [metrics_dict[n].get(metric, 0) for n in model_names]
            bars = ax.bar(model_names, values, color=colors)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=30)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("模型对比图已保存: %s", path)

    def _plot_feature_importance(self, df: pd.DataFrame):
        """特征重要性 Top-30"""
        fig, ax = plt.subplots(figsize=(10, 10))
        top = df.head(30)

        importance_col = [c for c in top.columns if "importance" in c.lower()][0]
        top_plot = top.iloc[::-1]  # 水平条形图从小到上

        ax.barh(range(len(top_plot)), top_plot[importance_col].values, color=sns.color_palette("viridis", len(top_plot)))
        ax.set_yticks(range(len(top_plot)))
        ax.set_yticklabels(top_plot["feature"].values[: len(top_plot)])
        ax.set_xlabel(importance_col.replace("_", " ").title())
        ax.set_title("Top 30 Feature Importance")

        path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("特征重要性图已保存: %s", path)

    def _plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray):
        """多分类 ROC 曲线"""
        if y_prob.ndim != 2 or y_prob.shape[1] != 3:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        classes = ["benign", "malicious", "suspicious"]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]

        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=[0, 1, 2])

        for i, (cls_name, color) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (OvR)")
        ax.legend(loc="lower right")

        path = os.path.join(self.output_dir, "roc_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("ROC 曲线已保存: %s", path)
