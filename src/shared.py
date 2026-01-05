import csv
import numpy as np
import numpy.typing as npt

INPUT_FILE: str = "../res/svmData.csv"
RES_DIR: str = "../out/res"

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]


def load_multiclass_csv(path: str) -> tuple[ArrayF, ArrayI]:
    """
    Load a classification dataset from a CSV file.
    Last column is the class label (-1 or +1).
    """
    data: list[list[float]] = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])

    data_np: ArrayF = np.asarray(data, dtype=np.float64)

    features: ArrayF = data_np[:, :-1]
    labels: ArrayI = data_np[:, -1].astype(np.int64)

    return features, labels
