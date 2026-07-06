"""
测试 FeatureEngineer fit/transform 一致性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config_loader import Config
from src.feature_engineer import FeatureEngineer


def make_df(n=200):
    """构造测试 DataFrame，列名与 config.yaml 的 columns.text_col 对齐"""
    np.random.seed(42)
    df = pd.DataFrame({
        "event_id": [f"evt_{i}" for i in range(n)],
        "message_sanitized": [f"some text data for item {i} " * ((i % 5) + 2) for i in range(n)],
        "num_a": np.random.randn(n) * 10,
        "num_b": np.random.randn(n) * 5,
        "cat_low": np.random.choice(["X", "Y", "Z"], n),
        "cat_high": [f"user_{i % 100}" for i in range(n)],
    })
    return df


def test_fit_transform_columns_match():
    """测试 fit/transform 后列名一致"""
    df = make_df(200)
    y = np.random.choice([0, 1, 2], 200)
    cfg = Config()
    fe = FeatureEngineer(cfg)

    fe.fit(df, y)
    X_t, _, _ = fe.transform(df)

    assert X_t.shape[1] == len(fe.all_feature_names)
    no_nan = X_t.isna().sum().sum()
    assert no_nan == 0, f"transform 后有 {no_nan} 个 NaN"


def test_interaction_column_stability():
    """核心测试: fit 时记录的交互列, transform 必须复用"""
    df_train = make_df(200)
    df_test = make_df(50)
    y = np.random.choice([0, 1, 2], 200)
    cfg = Config()
    fe = FeatureEngineer(cfg)

    fe.fit(df_train, y)
    X_t1, _, _ = fe.transform(df_train)
    X_t2, _, _ = fe.transform(df_test)

    assert set(X_t1.columns) == set(X_t2.columns), (
        f"fit/transform 列不一致: train={X_t1.shape} test={X_t2.shape}"
    )


def test_tfidf_consistency():
    """TF-IDF 特征数量应一致"""
    df = make_df(200)
    y = np.random.choice([0, 1, 2], 200)
    cfg = Config()

    # 确保文本特征启用
    cfg["features"]["text_feature"]["enabled"] = True

    fe = FeatureEngineer(cfg)
    fe.fit(df, y)
    X, _, _ = fe.transform(df)

    tfidf_cols = [c for c in fe.all_feature_names if c.startswith("tfidf_")]
    assert len(tfidf_cols) > 0, f"应该有 TF-IDF 特征，text_col={fe.text_col}"


def test_missing_feature_in_test():
    """测试集缺少某数值列时，scaler 也应补 0 列"""
    df_train = make_df(200)
    df_test = make_df(50).drop(columns=["num_b"])
    y = np.random.choice([0, 1, 2], 200)
    cfg = Config()
    fe = FeatureEngineer(cfg)

    fe.fit(df_train, y)
    X_test, _, _ = fe.transform(df_test)

    expected_cols = len(fe.all_feature_names)
    assert X_test.shape[1] == expected_cols, (
        f"expected {expected_cols} cols, got {X_test.shape[1]}. "
        f"test cols: {list(X_test.columns)}, expected: {fe.all_feature_names}"
    )


def test_save_load_roundtrip():
    """保存加载往返"""
    import tempfile, os
    df = make_df(200)
    y = np.random.choice([0, 1, 2], 200)
    cfg = Config()
    fe = FeatureEngineer(cfg)
    fe.fit(df, y)
    X_orig, _, _ = fe.transform(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        fe.save(tmpdir)
        fe2 = FeatureEngineer.load(tmpdir)
        X_loaded, _, _ = fe2.transform(df)

        pd.testing.assert_frame_equal(X_orig, X_loaded)


if __name__ == "__main__":
    test_fit_transform_columns_match()
    test_interaction_column_stability()
    test_tfidf_consistency()
    test_missing_feature_in_test()
    test_save_load_roundtrip()
    print("All feature_engineer tests passed!")
