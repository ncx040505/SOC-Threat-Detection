"""
测试模型模块
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.config_loader import Config
from src.models import ModelTrainer, IncrementalTrainer, compute_metrics


def make_data(n=100, n_features=10, n_classes=3):
    np.random.seed(42)
    X = np.random.randn(n, n_features)
    y = np.random.choice(range(n_classes), n)
    return X, y


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    y_prob = np.zeros((6, 3))
    y_prob[np.arange(6), y_pred] = 0.8
    y_prob[np.arange(6), y_true] = 0.1
    for i in range(6):
        y_prob[i, y_pred[i]] = 0.8

    m = compute_metrics(y_true, y_pred, y_prob)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1_macro"] <= 1
    assert "f1_malicious" in m


def test_build_models():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    models = trainer.build_models()
    assert len(models) >= 1


def test_train_single():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    trainer.build_models()
    X, y = make_data(100, 10)
    X_val, y_val = make_data(30, 10)

    for name, model in trainer.models.items():
        metrics = trainer.train_single(name, model, X, y, X_val, y_val)
        assert "accuracy" in metrics


def test_select_best():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    trainer.build_models()
    X, y = make_data(100, 10)
    X_val, y_val = make_data(30, 10)

    for name, model in trainer.models.items():
        trainer.train_single(name, model, X, y, X_val, y_val)

    best_name, best_model = trainer.select_best()
    assert best_name is not None


def test_ensemble():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    trainer.build_models()
    X, y = make_data(100, 10)
    X_val, y_val = make_data(30, 10)

    for name, model in trainer.models.items():
        trainer.train_single(name, model, X, y, X_val, y_val)

    trainer.build_ensemble()
    m = trainer.train_ensemble(X, y, X_val, y_val)
    if m:
        assert "accuracy" in m


def test_stacking():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    trainer.build_models()
    X, y = make_data(100, 10)
    X_val, y_val = make_data(30, 10)

    for name, model in trainer.models.items():
        trainer.train_single(name, model, X, y, X_val, y_val)

    trainer.build_stacking()
    m = trainer.train_stacking(X, y, X_val, y_val)
    if m:
        assert "accuracy" in m


def test_incremental():
    cfg = Config()
    cfg["incremental"]["enabled"] = True
    X, y = make_data(200, 10, 3)
    inc = IncrementalTrainer(cfg, n_classes=3, n_features=10)
    inc.build()
    inc.train(X, y)
    y_pred = inc.predict(X[:10])
    assert len(y_pred) == 10


def test_predict():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    trainer.build_models()
    X, y = make_data(100, 10)
    X_val, y_val = make_data(30, 10)

    for name, model in trainer.models.items():
        trainer.train_single(name, model, X, y, X_val, y_val)

    y_pred, y_prob = trainer.predict(X_val, use_ensemble=True)
    assert len(y_pred) == len(X_val)


def test_save_load_models():
    cfg = Config()
    trainer = ModelTrainer(cfg)
    trainer.build_models()
    X, y = make_data(100, 10)
    X_val, y_val = make_data(30, 10)

    for name, model in trainer.models.items():
        trainer.train_single(name, model, X, y, X_val, y_val)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save(tmpdir)
        trainer2 = ModelTrainer(cfg)
        trainer2.load(tmpdir)
        assert len(trainer2.trained_models) == len(trainer.trained_models)


if __name__ == "__main__":
    test_compute_metrics()
    test_build_models()
    test_train_single()
    test_select_best()
    test_ensemble()
    test_stacking()
    test_incremental()
    test_predict()
    test_save_load_models()
    print("All models tests passed!")
