from src.data_utils import load_csv
from src.ensemble_experiments import rf_experiment, gb_experiment, plot_feature_importance

if __name__ == "__main__":
    features, labels = load_csv("../res/data_2.csv")

    rf_experiment(features, labels)
    gb_experiment(features, labels)
    plot_feature_importance(features, labels)
