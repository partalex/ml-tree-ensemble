from src.data_utils import load_csv
from src.decision_tree_experiments import train_tree, plot_decision_boundary, plot_tree_structure
from src.feature_selection import correlation_ranking, plot_correlation_ranking, wrapper_feature_selection, \
    plot_wrapper_scores

if __name__ == "__main__":
    features, labels = load_csv("../res/data_1.csv")

    # 1. Correlation
    ranking = correlation_ranking(features, labels)
    plot_correlation_ranking(ranking)

    # 2. Wrapper feature selection
    selected = wrapper_feature_selection(features, labels)
    print("Chosen features (wrapper):", selected)
    plot_wrapper_scores(features, labels)

    # 3. Two predictors for the tree
    feature_pair = features[:, selected[:2]]

    # 4. Underfitted
    clf_under = train_tree(feature_pair, labels, max_depth=1)
    plot_decision_boundary(clf_under, feature_pair, labels, "Underfitted model")

    # 5. Well-fitted
    clf_good = train_tree(feature_pair, labels, max_depth=4)
    plot_decision_boundary(clf_good, feature_pair, labels, "Well-fitted model")

    # 6. Overfitted
    clf_over = train_tree(feature_pair, labels, max_depth=20)
    plot_decision_boundary(clf_over, feature_pair, labels, "Overfitted model")

    plot_tree_structure(clf_good)
