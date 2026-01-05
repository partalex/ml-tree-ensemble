import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def correlation_ranking(
        features: np.ndarray,
        labels: np.ndarray,
) -> list[tuple[int, float]]:
    """
    Rank features by absolute Pearson correlation with target.
    Args:
        features: 2D array of shape (n_samples, n_features)
        labels: 1D array of shape (n_samples)
    Returns:
        List of tuples (feature_index, abs_correlation), sorted by abs_correlation descending.
    """
    correlations: list[tuple[int, float]] = []
    for i in range(features.shape[1]):
        corr: float = np.corrcoef(features[:, i], labels)[0, 1]
        correlations.append((i, abs(corr)))

    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations


def plot_correlation_ranking(
        ranking: list[tuple[int, float]]
) -> None:
    """
    Plot bar chart of feature correlation ranking.
    Args:
        ranking: List of tuples (feature_index, abs_correlation)
    """
    indices, values = zip(*ranking)
    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xlabel("Feature (ranked)")
    plt.ylabel("|correlation|")
    plt.title("Feature ranking by correlation")
    plt.show()


def wrapper_feature_selection(
        features: np.ndarray,
        labels: np.ndarray,
) -> list[int]:
    """
    Wrapper method using Sequential Forward Selection.
    Args:
        features: 2D array of shape (n_samples, n_features)
        labels: 1D array of shape (n_samples)
    Returns:
        List of selected feature indices.
    """
    model = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=5)

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select="auto",
        direction="forward",
        cv=cv,
        scoring="accuracy",
    )

    sfs.fit(features, labels)
    selected: list[int] = list(sfs.get_support(indices=True))
    return selected


def plot_wrapper_scores(
        features: np.ndarray,
        labels: np.ndarray,
) -> None:
    """
    Accuracy vs number of selected features.
    Args:
        features: 2D array of shape (n_samples, n_features)
        labels: 1D array of shape (n_samples)
    """
    model = LogisticRegression(max_iter=2000)
    scores: list[float] = []

    for k in range(1, features.shape[1] + 1):
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select=k,
            direction="forward",
            cv=5,
            scoring="accuracy",
        )
        sfs.fit(features, labels)
        scores.append(sfs.fit(features, labels))

    plt.figure()
    plt.plot(range(1, features.shape[1] + 1), scores, marker="o")
    plt.xlabel("Number of predictors")
    plt.ylabel("Accuracy")
    plt.title("Wrapper method – quality depending on number of predictors")
    plt.show()
