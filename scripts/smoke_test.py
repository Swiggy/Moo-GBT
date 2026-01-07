"""Smoke tests for Moo-GBT classifier and regressor."""

import numpy as np
from sklearn.datasets import load_iris, load_diabetes

from multiobjective_gbt import MooGBTClassifier, MooGBTRegressor


def run_classifier_smoke():
    X, y = load_iris(return_X_y=True)
    # Use a binary slice to avoid abstract multinomial loss implementation
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
    temp = clf.calibrate_temperature(X[:20], y[:20])
    fi = clf.compute_feature_importances()
    shap_trees = clf.get_shap_trees()
    print("Classifier proba shape", proba.shape)
    print("Classifier temperature", temp)
    print("Classifier feature_importances shape", fi.shape)
    print("Classifier shap trees", len(shap_trees))


def run_regressor_smoke():
    X, y = load_diabetes(return_X_y=True)
    reg = MooGBTRegressor(
        n_estimators=5,
        random_state=0,
        early_stopping_patience=2,
    )
    reg.fit(X, y)
    preds = reg.predict(X[:5])
    fi = reg.compute_feature_importances()
    shap_trees = reg.get_shap_trees()
    print("Regressor preds shape", preds.shape)
    print("Regressor feature_importances shape", fi.shape)
    print("Regressor shap trees", len(shap_trees))


def main():
    run_classifier_smoke()
    run_regressor_smoke()


if __name__ == "__main__":
    main()
