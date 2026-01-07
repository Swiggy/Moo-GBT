import numpy as np
from sklearn.datasets import load_iris, load_diabetes

from multiobjective_gbt import MooGBTClassifier, MooGBTRegressor


def test_classifier_fit_predict_proba():
    X, y = load_iris(return_X_y=True)
    # Binary slice for compatibility with binomial loss
    mask = y < 2
    X = X[mask]
    y = y[mask]

    clf = MooGBTClassifier(
        n_estimators=5,
        random_state=0,
        early_stopping_patience=2,
        early_stopping_metric="train_loss",
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X[:5])
    assert isinstance(proba, np.ndarray)
    assert proba.shape[0] == 5

    temp = clf.calibrate_temperature(X[:20], y[:20])
    assert isinstance(float(temp), float)

    fi = clf.compute_feature_importances()
    assert isinstance(fi, np.ndarray)
    assert fi.shape[0] == X.shape[1]

    shap_trees = clf.get_shap_trees()
    assert shap_trees is not None


def test_regressor_fit_predict():
    X, y = load_diabetes(return_X_y=True)

    reg = MooGBTRegressor(
        n_estimators=5,
        random_state=0,
        early_stopping_patience=2,
    )
    reg.fit(X, y)

    preds = reg.predict(X[:5])
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 5

    fi = reg.compute_feature_importances()
    assert isinstance(fi, np.ndarray)
    assert fi.shape[0] == X.shape[1]

    shap_trees = reg.get_shap_trees()
    assert shap_trees is not None
