import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_tree(
        features: np.ndarray,
        labels: np.ndarray,
        max_depth: int | None = None,
) -> DecisionTreeClassifier:
    """
    Train a decision tree classifier.
    Args:
        features (np.ndarray): The input features for training.
        labels (np.ndarray): The target labels for training.
        max_depth (int | None): The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    Returns:
        DecisionTreeClassifier: The trained decision tree classifier.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(features, labels)
    return clf


def plot_decision_boundary(
        clf: DecisionTreeClassifier,
        features: np.ndarray,
        labels: np.ndarray,
        title: str,
        out_path: str,
) -> None:
    """
    Plot the decision boundary of a classifier.
    Args:
        clf (DecisionTreeClassifier): The trained classifier.
        features (np.ndarray): The input features.
        labels (np.ndarray): The target labels.
        title (str): The title of the plot.
        out_path (str): Path to save the plot. If None, only shows the plot.
    """
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    predictions = clf.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
    predictions = predictions.reshape(x_grid.shape)

    plt.figure()
    plt.contourf(x_grid, y_grid, predictions, alpha=0.3)
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolor="k")
    plt.title(title)
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_tree_structure(
        clf: DecisionTreeClassifier,
        out_path: str,
) -> None:
    """
    Plot the structure of a decision tree.
    Args:
        clf (DecisionTreeClassifier): The trained decision tree classifier.
        out_path (str): Path to save the plot. If None, only shows the plot.
    """
    plt.figure(figsize=(16, 8))
    plot_tree(clf, filled=True)
    plt.savefig(out_path, dpi=300)
    plt.show()
