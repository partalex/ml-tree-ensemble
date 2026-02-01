from shared import DATA_2_PATH, OUT_2_DIR
from src.data_utils import load_csv
from src.ensemble_experiments import rf_experiment, gb_experiment, plot_feature_importance

if __name__ == "__main__":
    features, labels = load_csv(DATA_2_PATH)

    # 1. Random Forest Experiment
    rf_experiment(
        features,
        labels,
        f"{OUT_2_DIR}/rf_n_estimators.png",
    )

    # 2. Gradient Boosting Experiment
    gb_experiment(
        features,
        labels,
        f"{OUT_2_DIR}/gn_learning_rate.png",
    )

    # 3. Feature Importance Plotting
    plot_feature_importance(
        features,
        labels,
        f"{OUT_2_DIR}/feature_importances.png"
    )
