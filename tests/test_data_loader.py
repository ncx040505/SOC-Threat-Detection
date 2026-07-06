"""
测试 DataLoader fit/transform 一致性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config_loader import Config, Config as _Cfg
from src.data_loader import DataLoader


def make_train_df(n=200):
    """构造一个假的训练集"""
    np.random.seed(42)
    df = pd.DataFrame({
        "event_id": [f"evt_{i}" for i in range(n)],
        "raw_message": [f"msg_{i}" * 5 for i in range(n)],
        "num_feat": np.random.randn(n),
        "num_feat_missing": np.random.randn(n),
        "cat_feat": np.random.choice(["A", "B", "C", "D", "E"], n),
        "high_card_feat": [f"h_{i % 50}" for i in range(n)],
        "dense_cat": np.random.choice([0, 1, 2], n),
        "label": np.random.choice(["benign", "malicious", "suspicious"], n),
    })
    df.loc[np.random.choice(n, 30, replace=False), "num_feat_missing"] = np.nan
    return df


def test_fit_transform_consistency():
    """测试：fit 后的 transform 输出列一致"""
    df_train = make_train_df(200)
    cfg = Config()
    dl = DataLoader(cfg)

    dl.fit(df_train)
    X, y, eids = dl.transform(df_train, has_label=True)

    assert not X.empty
    assert len(X.columns) > 0
    assert set(X.columns) == set(dl.feature_order)
    assert X.isna().sum().sum() == 0, "transform 后不应有 NaN"


def test_transform_new_data():
    """测试：新数据 transform 输出的列与训练时一致"""
    df_train = make_train_df(200)
    df_test = make_train_df(50)  # 假装从另一个分布来
    cfg = Config()
    dl = DataLoader(cfg)

    dl.fit(df_train)
    X_tr, _, _ = dl.transform(df_train)
    X_te, _, _ = dl.transform(df_test, has_label=False)

    # 列应该相同
    assert set(X_tr.columns) == set(X_te.columns), (
        f"train cols {set(X_tr.columns)} vs test cols {set(X_te.columns)}"
    )


def test_missing_column_in_test():
    """测试：测试集缺少某列时补 0"""
    df_train = make_train_df(200)
    df_test = make_train_df(50).drop(columns=["num_feat_missing"])
    cfg = Config()
    dl = DataLoader(cfg)

    dl.fit(df_train)
    X_tr, _, _ = dl.transform(df_train)
    X_te, _, _ = dl.transform(df_test, has_label=False)

    assert "num_feat_missing" in X_te.columns
    assert X_te["num_feat_missing"].sum() == 0.0  # 新数据中全为 0


def test_extra_column_in_test():
    """测试：测试集多出列时丢弃"""
    df_train = make_train_df(200)
    df_test = make_train_df(50)
    df_test["extra_col"] = 123
    cfg = Config()
    dl = DataLoader(cfg)

    dl.fit(df_train)
    X_te, _, _ = dl.transform(df_test, has_label=False)

    assert "extra_col" not in X_te.columns


def test_label_encoding():
    """测试：标签编码正确"""
    df_train = make_train_df(200)
    cfg = Config()
    dl = DataLoader(cfg)

    dl.fit(df_train)
    assert set(dl.label_encoder.keys()) == {"benign", "malicious", "suspicious"}
    assert dl.label_decoder[0] == "benign"


def test_high_missing_col_dropped():
    """测试：高缺失率列被记录丢弃"""
    df_train = make_train_df(200)
    # 制造一个 90% 缺失的列
    df_train["mostly_missing"] = np.random.choice([np.nan, 1.0], 200, p=[0.9, 0.1])
    cfg = Config()
    # 缺失阈值 0.6
    cfg_data = dict(cfg.setdefault("data", {}))
    cfg_data["missing_threshold"] = 0.6
    cfg["data"] = cfg_data

    dl = DataLoader(cfg)
    dl.fit(df_train)
    assert "mostly_missing" in dl.drop_cols


def test_save_load_roundtrip():
    """测试：保存加载往返"""
    import tempfile, os
    df_train = make_train_df(200)
    cfg = Config()
    dl = DataLoader(cfg)
    dl.fit(df_train)
    X_orig, _, _ = dl.transform(df_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        dl.save(tmpdir)
        dl2 = DataLoader.load(cfg, tmpdir)
        X_loaded, _, _ = dl2.transform(df_train)

        pd.testing.assert_frame_equal(X_orig, X_loaded)


if __name__ == "__main__":
    test_fit_transform_consistency()
    test_transform_new_data()
    test_missing_column_in_test()
    test_extra_column_in_test()
    test_label_encoding()
    test_high_missing_col_dropped()
    test_save_load_roundtrip()
    print("All data_loader tests passed!")
