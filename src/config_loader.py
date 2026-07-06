"""
配置加载器 — 单例模式加载 config.yaml
✅ v2.0: 新增 columns 映射、stacking、hyperopt、incremental、feature_selection
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """支持属性访问的字典"""

    def __getattr__(self, key: str) -> Any:
        if key in self:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, AttrDict):
                val = AttrDict(val)
                self[key] = val
            return val
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value


class Config(AttrDict):
    """全局单例配置"""

    _instance: Optional["Config"] = None
    _path: Optional[str] = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._load_config(config_path)
            cls._instance = instance
        elif config_path is not None and config_path != cls._path:
            # 重新加载
            instance = cls._instance
            instance.clear()
            instance._load_config(config_path)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        # All init happens in __new__; prevent dict.__init__ from interpreting config_path
        pass

    def _load_config(self, config_path: Optional[str] = None):
        if config_path is None:
            # 搜索默认配置位置（优先工作目录，其次 package 目录）
            candidates = [
                "configs/config.yaml",
                Path(__file__).resolve().parent.parent / "configs" / "config.yaml",
            ]
            for c in candidates:
                if os.path.exists(c):
                    config_path = c
                    break
            if config_path is None:
                raise FileNotFoundError("找不到 config.yaml，请通过 --config 指定路径")

        Config._path = config_path
        logger.info("加载配置: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.clear()
        self.update(data)

        # 递归转换嵌套字典
        self._convert_dicts(self)

        # 确保关键路径为绝对路径
        project_root = Path(config_path).resolve().parent
        for key in ("raw_dir", "processed_dir", "split_dir", "model_dir", "output_dir", "viz_dir", "log_dir"):
            if "paths" in self and key in self.paths:
                self.paths[key] = os.path.join(project_root, self.paths[key])

    def _convert_dicts(self, d: dict):
        """递归转换所有嵌套字典为 AttrDict"""
        for k, v in list(d.items()):
            if isinstance(v, dict):
                d[k] = AttrDict(v)
                self._convert_dicts(d[k])
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        v[i] = AttrDict(item)
                        self._convert_dicts(v[i])

    @classmethod
    def reset(cls):
        """重置单例（主要用于测试）"""
        cls._instance = None
        cls._path = None
