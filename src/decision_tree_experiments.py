import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_tree(
        features: np.ndarray,
        labels: np.ndarray,
        max_depth: int,
) -> DecisionTreeClassifier:
    """
    Train a decision tree classifier.
    Args:
        features (np.ndarray): The input features for training.
        labels (np.ndarray): The target labels for training.
        max_depth (int): The maximum depth of the decision tree.
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
) -> None:
    """
    Plot the decision boundary of a classifier.
    Args:
        clf (DecisionTreeClassifier): The trained classifier.
        features (np.ndarray): The input features.
        labels (np.ndarray): The target labels.
        title (str): The title of the plot.
    """
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, predictions, alpha=0.3)
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolor="k")
    plt.title(title)
    plt.show()


def plot_tree_structure(clf: DecisionTreeClassifier) -> None:
    """
    Plot the structure of a decision tree.
    Args:
        clf (DecisionTreeClassifier): The trained decision tree classifier.
    """
    plt.figure(figsize=(16, 8))
    plot_tree(clf, filled=True)
    plt.show()
