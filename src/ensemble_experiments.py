import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def rf_experiment(
        features: np.ndarray,
        labels: np.ndarray,
        out_path: str,
) -> None:
    """
    Plots the effect of the number of estimators on Random Forest classifier performance.
    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Target labels.
        out_path (str): Path to save the plot.
    """
    n_estimators_list: list[int] = [10, 50, 100, 200]
    scores: list[float] = []

    for n in n_estimators_list:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=5,
            max_features="sqrt",
        )
        score = cross_val_score(rf, features, labels, cv=5).mean()
        scores.append(score)

    plt.figure()
    plt.plot(n_estimators_list, scores, marker="o")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("Random Forest - Ensemble Size Impact")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()


def gb_experiment(
        features: np.ndarray,
        labels: np.ndarray,
        out_path: str,
) -> None:
    """
    Plots the effect of learning rate on Gradient Boosting classifier performance.
    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Target labels.
        out_path (str): Path to save the plot.
    """
    learning_rates: list[float] = [0.01, 0.05, 0.1, 0.2]
    scores: list[float] = []

    for lr in learning_rates:
        gb = GradientBoostingClassifier(
            learning_rate=lr,
            n_estimators=100,
            max_depth=3,
        )
        score = cross_val_score(gb, features, labels, cv=5).mean()
        scores.append(score)

    plt.figure()
    plt.plot(learning_rates, scores, marker="o")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Gradient Boosting – Learning Rate Impact")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(
        features: np.ndarray,
        labels: np.ndarray,
        out_path: str,
) -> None:
    """
    Plots feature importances using a Random Forest classifier.
    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Target labels.
        out_path (str): Path to save the plot.
    """
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(features, labels)

    importances = rf.feature_importances_

    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances (Random Forest)")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
