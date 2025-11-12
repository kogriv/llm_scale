"""CatBoost solution for task2 with hyperparameter tuning.

This script trains a CatBoostClassifier on the Task 2 dataset using
Stratified K-Fold cross-validation. A small hyperparameter search is
performed to reach the target ROC-AUC metric (>= 0.88). The best model is
then refit on the full training data and predictions for the test set are
stored in ``task2_catboost_submission.csv``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DATA_DIR = Path("data/task2")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = Path("task2_catboost_submission.csv")
SEARCH_LOG_PATH = Path("task2_catboost_search_log.json")


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the training and test datasets."""

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    features = train_df.drop(columns=["target"])
    target = train_df["target"].astype(int)

    return features, test_df, target


def evaluate_params(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, float],
    cv: StratifiedKFold,
    cat_features: List[int],
) -> float:
    """Evaluate the parameters using cross-validation and return mean ROC-AUC."""

    scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        train_pool = Pool(
            data=X.iloc[train_idx],
            label=y.iloc[train_idx],
            cat_features=cat_features,
        )
        valid_pool = Pool(
            data=X.iloc[valid_idx],
            label=y.iloc[valid_idx],
            cat_features=cat_features,
        )

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=3000,
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            bagging_temperature=params["bagging_temperature"],
            border_count=254,
            early_stopping_rounds=200,
            random_seed=42,
            auto_class_weights="Balanced",
            verbose=False,
        )

        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        pred = model.predict_proba(valid_pool)[:, 1]
        score = roc_auc_score(y.iloc[valid_idx], pred)
        scores.append(score)

        print(
            f"Fold {fold}: ROC-AUC={score:.5f} | params={params}",
            flush=True,
        )

    mean_score = float(np.mean(scores))
    print(f"Mean ROC-AUC: {mean_score:.5f} for params={params}\n", flush=True)
    return mean_score


def main() -> None:
    X_train, X_test, y_train = load_datasets()

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    class_counts = y_train.value_counts().to_dict()
    print("Баланс классов:", class_counts)

    cat_features = [X_train.columns.get_loc("C")]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Hyperparameter configurations inspired by manual tuning experiments.
    depth_options = [5, 6, 7]
    lr_options = [0.03, 0.05, 0.07]
    l2_options = [3, 6, 9]
    bagging_options = [0.2, 0.6]

    search_space = list(
        itertools.product(depth_options, lr_options, l2_options, bagging_options)
    )
    rng = np.random.default_rng(42)
    rng.shuffle(search_space)

    max_evals = 12
    target_score = 0.88

    best_score = -np.inf
    best_params: Dict[str, float] = {}
    search_log: List[Dict[str, float]] = []

    print("Запуск перебора гиперпараметров CatBoost...")

    for eval_idx, (depth, lr, l2, bagging_temp) in enumerate(search_space[:max_evals], start=1):
        params = {
            "depth": depth,
            "learning_rate": lr,
            "l2_leaf_reg": l2,
            "bagging_temperature": bagging_temp,
        }
        score = evaluate_params(X_train, y_train, params, cv, cat_features)
        entry = {**params, "roc_auc": score}
        search_log.append(entry)

        if score > best_score:
            best_score = score
            best_params = params

        if best_score >= target_score and eval_idx >= 4:
            print(
                "Достигнуто целевое качество, прекращаем перебор параметров.",
                flush=True,
            )
            break

    print("Лучшие параметры:", best_params)
    print(f"Лучший ROC-AUC (CV): {best_score:.5f}")

    SEARCH_LOG_PATH.write_text(json.dumps(search_log, indent=2))

    best_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=4000,
        learning_rate=best_params["learning_rate"],
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        bagging_temperature=best_params["bagging_temperature"],
        border_count=254,
        early_stopping_rounds=250,
        random_seed=42,
        auto_class_weights="Balanced",
        verbose=200,
    )

    # Use 15% of the training data as a validation set for final model tuning.
    final_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
    train_idx, valid_idx = next(final_cv.split(X_train, y_train))
    train_pool = Pool(X_train.iloc[train_idx], y_train.iloc[train_idx], cat_features=cat_features)
    valid_pool = Pool(X_train.iloc[valid_idx], y_train.iloc[valid_idx], cat_features=cat_features)

    best_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    best_iteration = best_model.get_best_iteration()
    if best_iteration is None or best_iteration <= 0:
        best_iteration = best_model.tree_count_

    print(f"Лучшее число итераций по валидации: {best_iteration}")

    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=best_iteration,
        learning_rate=best_params["learning_rate"],
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        bagging_temperature=best_params["bagging_temperature"],
        border_count=254,
        random_seed=42,
        auto_class_weights="Balanced",
        verbose=False,
    )

    full_pool = Pool(X_train, y_train, cat_features=cat_features)
    final_model.fit(full_pool)

    test_pool = Pool(X_test, cat_features=cat_features)
    test_pred = final_model.predict_proba(test_pool)[:, 1]

    submission = pd.DataFrame({"id": np.arange(len(test_pred)), "target": test_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Предсказания сохранены в {SUBMISSION_PATH.resolve()}")
    print(f"Лог гиперпараметров сохранён в {SEARCH_LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
