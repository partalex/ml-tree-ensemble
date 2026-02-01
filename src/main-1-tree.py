from shared import DATA_1_PATH, OUT_1_DIR
from src.data_utils import load_csv
from src.decision_tree_experiments import train_tree, plot_decision_boundary, plot_tree_structure
from src.feature_selection import (
    correlation_ranking,
    plot_correlation_ranking,
    plot_wrapper_scores,
)

if __name__ == "__main__":
    features, labels = load_csv(DATA_1_PATH)

    # 1. Correlation ranking
    ranking = correlation_ranking(features, labels)
    plot_correlation_ranking(ranking, f"{OUT_1_DIR}/correlation_ranking.png")

    # 2. Wrapper ranking (Logistic Regression + CV accuracy)
    plot_wrapper_scores(features, labels, out_path=f"{OUT_1_DIR}/wrapper_scores.png")

    # 3. Take TOP-2 predictors by correlation for the tree
    pair_idx = [ranking[0][0], ranking[1][0]]
    feature_pair = features[:, pair_idx]

    print("Using feature pair (from correlation ranking):", pair_idx)

    # 4. Underfitted
    clf_under = train_tree(feature_pair, labels, max_depth=1)
    plot_decision_boundary(
        clf_under,
        feature_pair,
        labels,
        "Underfitted model",
        f"{OUT_1_DIR}/underfitted_decision_boundary.png",
    )
    plot_tree_structure(
        clf_under,
        f"{OUT_1_DIR}/underfitted_tree_structure.png",
    )

    # 5. Well-fitted
    clf_good = train_tree(feature_pair, labels, max_depth=4)
    plot_decision_boundary(
        clf_good,
        feature_pair,
        labels,
        "Well-fitted model",
        f"{OUT_1_DIR}/well_fitted_decision_boundary.png",
    )
    plot_tree_structure(clf_good, f"{OUT_1_DIR}/well_fitted_tree_structure.png")

    # 6. Overfitted
    clf_over = train_tree(feature_pair, labels)
    plot_decision_boundary(
        clf_over,
        feature_pair,
        labels,
        "Overfitted model",
        f"{OUT_1_DIR}/overfitted_decision_boundary.png",
    )
    plot_tree_structure(clf_over, f"{OUT_1_DIR}/overfitted_tree_structure.png")
