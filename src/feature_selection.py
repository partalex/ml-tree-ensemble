import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def correlation_ranking(
        features: np.ndarray,
        labels: np.ndarray,
) -> list[tuple[int, float]]:
    """
    Filter method:
    Rank features by absolute Pearson correlation with target variables.
    Args:
        features: feature matrix (n_samples x n_features)
        labels: target variable vector (n_samples)
    """
    correlations: list[tuple[int, float]] = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], labels)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations.append((i, float(abs(corr))))

    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations


def plot_correlation_ranking(
        ranking: list[tuple[int, float]],
        out_path: str,
) -> None:
    """
    Plot feature ranking by absolute Pearson correlation with
    target variables.
    Args:
        ranking: list of (feature index, |correlation|) tuples
        out_path: path to save the plot
    """
    indices, values = zip(*ranking)
    feature_names = [f"P_{i}" for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, values)
    plt.gca().invert_yaxis()
    plt.xlabel("|correlation|")
    plt.ylabel("Feature")
    plt.title("Feature ranking by absolute Pearson correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_wrapper_scores(
        features: np.ndarray,
        labels: np.ndarray,
        out_path: str,
) -> None:
    """
    Wrapper method:
    Plot feature ranking by cross-validated accuracy of Logistic Regression
    using each feature individually.
    Args:
        features: feature matrix (n_samples x n_features)
        labels: target variable vector (n_samples,)
        out_path: path to save the plot (if None, show it)
    """
    model = LogisticRegression(max_iter=4000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for i in range(features.shape[1]):
        feature_one = features[:, i:i + 1]  # only one feature
        acc = cross_val_score(model, feature_one, labels, cv=cv, scoring="accuracy").mean()
        scores.append((i, float(acc)))

    # sort descending by accuracy
    scores.sort(key=lambda x: x[1], reverse=True)

    indices = [i for i, _ in scores]
    values = [v for _, v in scores]
    names = [f"P_{i}" for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.xlabel("Cross-validated accuracy (Logistic Regression, single feature)")
    plt.ylabel("Feature")
    plt.title("Feature Ranking by Wrapper Method (LR accuracy per feature)")
    plt.grid(True, axis="x", alpha=0.25)

    # scale to be "tight" (like in your example)
    min_accuracy, max_accuracy = min(values), max(values)
    pad = max(0.001, 0.02 * (max_accuracy - min_accuracy if max_accuracy > min_accuracy else 1.0))
    plt.xlim(min_accuracy - pad, max_accuracy + pad)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
