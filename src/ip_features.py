"""
IP 行为统计特征引擎
========================
方案 C 核心组件: 从 IP 地址中提取行为指纹，不依赖 Target Encoding

特征家族:
  1. IP 行为统计 (per src_ip / per dst_ip 聚合)
  2. IP 网络拓扑特征 (IP 前缀, 内网/外网, 网段)
  3. IP 行为指纹 (时间窗口内的活动模式)
"""
from __future__ import annotations

import ipaddress
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class IPBehaviorFeatures(BaseEstimator, TransformerMixin):
    """
    IP 行为统计特征:
    - 对 src_ip 计算: 关联的 dst_ip 去重数, dst_port 去重数, src_port 去重数,
      product_name 去重数, pipeline 去重数, 时间跨度, 总记录数
    - 对 dst_ip 计算: 关联的 src_ip 去重数, 出现次数

    重要: fit 阶段从 train 数据中学习映射表, transform 阶段对 train 和 test 都适用
          对于 test 中 unseen 的 IP, 使用默认值 (如 1)
    """

    # 私有 IP 范围 (RFC 1918 + 载波级 NAT)
    _PRIVATE_NETS = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ]

    def __init__(
        self,
        ip_cols: Optional[List[str]] = None,
        compute_on_train_only: bool = True,
    ):
        self.ip_cols = ip_cols or ["src_ip", "dst_ip"]
        self.compute_on_train_only = compute_on_train_only

        # fit 阶段计算的聚合特征映射
        self._src_ip_stats: Dict[str, Dict[str, float]] = {}
        self._dst_ip_stats: Dict[str, Dict[str, float]] = {}
        self._default_src_stats: Dict[str, float] = {}
        self._default_dst_stats: Dict[str, float] = {}
        self.fitted = False

        # 记录列名
        self._feature_names_out: List[str] = []

    @staticmethod
    def _is_private_ip(ip_str: str) -> int:
        """判断是否为私有 IP"""
        try:
            ip = ipaddress.ip_address(ip_str)
            for net in IPBehaviorFeatures._PRIVATE_NETS:
                if ip in net:
                    return 1
            return 0
        except (ValueError, TypeError):
            return -1  # 无效 IP

    @staticmethod
    def _ip_to_prefix(ip_str: str, bits: int = 16) -> int:
        """将 IP 转为 /bits 前缀的数值表示 → 用作 one-hot 或直接做 ordinal"""
        try:
            ip = ipaddress.ip_address(ip_str)
            if isinstance(ip, ipaddress.IPv4Address):
                net = ipaddress.ip_network(f"{ip}/{bits}", strict=False)
                return int(net.network_address)
            return -1
        except (ValueError, TypeError):
            return -2

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        从 train 数据中预计算 IP 聚合统计
        """
        logger.info("IPBehaviorFeatures.fit — 从 %d 条记录中提取 IP 行为统计", len(X))

        # ---- 1. src_ip 聚合 ----
        if "src_ip" in X.columns and "src_host" not in X.columns:
            # 按 src_ip 分组
            src_grp = X.groupby("src_ip")

            # 需要哪些列做统计
            cols_available = [c for c in ["dst_ip", "src_port", "product_name", "pipeline", "timestamp"]
                              if c in X.columns]

            src_agg: Dict[str, pd.DataFrame] = {}

            if "dst_ip" in cols_available:
                src_agg["src_dst_ip_count"] = src_grp["dst_ip"].nunique()
            if "src_port" in cols_available:
                src_agg["src_port_count"] = src_grp["src_port"].nunique()
            if "product_name" in cols_available:
                src_agg["src_product_count"] = src_grp["product_name"].nunique()
            if "pipeline" in cols_available:
                src_agg["src_pipeline_count"] = src_grp["pipeline"].nunique()
            if "timestamp" in cols_available:
                src_agg["src_time_span"] = src_grp["timestamp"].agg(lambda x: x.max() - x.min())

            # 总记录数
            src_agg["src_total_events"] = src_grp.size()

            if src_agg:
                src_stats_df = pd.DataFrame(src_agg)
                self._src_ip_stats = src_stats_df.to_dict(orient="index")
                # 计算默认值 (取中位数)
                self._default_src_stats = {
                    k: src_stats_df[k].median() if not src_stats_df[k].empty else 1
                    for k in src_stats_df.columns
                }

        # ---- 2. dst_ip 聚合 ----
        if "dst_ip" in X.columns:
            dst_grp = X.groupby("dst_ip")
            dst_agg: Dict[str, pd.DataFrame] = {}

            if "src_ip" in X.columns:
                dst_agg["dst_src_ip_count"] = dst_grp["src_ip"].nunique()
            if "src_port" in X.columns:
                dst_agg["dst_port_count"] = dst_grp["src_port"].nunique()

            dst_agg["dst_total_events"] = dst_grp.size()

            if dst_agg:
                dst_stats_df = pd.DataFrame(dst_agg)
                self._dst_ip_stats = dst_stats_df.to_dict(orient="index")
                self._default_dst_stats = {
                    k: dst_stats_df[k].median() if not dst_stats_df[k].empty else 1
                    for k in dst_stats_df.columns
                }

        # ---- 3. 记录列名 ----
        self._feature_names_out = list(self._default_src_stats.keys()) + list(self._default_dst_stats.keys())
        logger.info("IPBehaviorFeatures.fit — 产出 %d 个 IP 行为特征", len(self._feature_names_out))

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """对每条记录附加其 src_ip / dst_ip 的行为统计特征"""
        if not self.fitted:
            raise RuntimeError("IPBehaviorFeatures 需要先 fit")

        result = pd.DataFrame(index=X.index)

        # src_ip 特征
        if self._src_ip_stats:
            src_default = pd.Series(self._default_src_stats)
            src_mapped = X["src_ip"].apply(
                lambda ip: pd.Series(self._src_ip_stats.get(ip, self._default_src_stats))
            )
            for col in self._default_src_stats:
                result[col] = src_mapped[col].values

        # dst_ip 特征
        if self._dst_ip_stats:
            dst_default = pd.Series(self._default_dst_stats)
            dst_mapped = X["dst_ip"].apply(
                lambda ip: pd.Series(self._dst_ip_stats.get(ip, self._default_dst_stats))
            )
            for col in self._default_dst_stats:
                result[col] = dst_mapped[col].values

        return result.values

    def get_feature_names_out(self) -> List[str]:
        return list(self._feature_names_out)


class IPTopologyFeatures(BaseEstimator, TransformerMixin):
    """
    IP 网络拓扑特征:
    - is_private_src, is_private_dst
    - is_same_subnet (src 和 dst 是否在同一 /24, /16)
    - src_ip_prefix_16, dst_ip_prefix_16 (整型表示, 用于 ordinal encoding)
    - src_ip_prefix_24, dst_ip_prefix_24
    """

    def __init__(self):
        self.fitted = False
        self._feature_names_out: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._feature_names_out = [
            "is_private_src",
            "is_private_dst",
            "is_same_subnet_24",
            "is_same_subnet_16",
        ]
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        result = pd.DataFrame(index=X.index)

        # 私有 IP
        result["is_private_src"] = X["src_ip"].apply(IPBehaviorFeatures._is_private_ip)
        result["is_private_dst"] = X["dst_ip"].apply(IPBehaviorFeatures._is_private_ip)

        # 同子网
        src_prefix_24 = X["src_ip"].apply(lambda x: IPBehaviorFeatures._ip_to_prefix(x, 24))
        dst_prefix_24 = X["dst_ip"].apply(lambda x: IPBehaviorFeatures._ip_to_prefix(x, 24))
        result["is_same_subnet_24"] = (src_prefix_24 == dst_prefix_24).astype(int)

        src_prefix_16 = X["src_ip"].apply(lambda x: IPBehaviorFeatures._ip_to_prefix(x, 16))
        dst_prefix_16 = X["dst_ip"].apply(lambda x: IPBehaviorFeatures._ip_to_prefix(x, 16))
        result["is_same_subnet_16"] = (src_prefix_16 == dst_prefix_16).astype(int)

        return result.values

    def get_feature_names_out(self) -> List[str]:
        return list(self._feature_names_out)
