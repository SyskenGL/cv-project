import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from cv.core.dexpression import DeXpression
from cv.dataset.loader import CKPLoader, MMILoader
from sklearn.metrics import confusion_matrix as cm


if __name__ == "__main__":

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    logging.basicConfig(
        format="\n%(asctime)s [%(levelname)-5.5s] \n%(message)s\n",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(save_path, "logs", f"{time.strftime('%Y%m%d-%H%M%S')}.log"),
                encoding="utf-8"
            )
        ]
    )

    # Loading Dataset
    choice = None
    while choice not in ["0", "1", "2"]:
        choice = input("\n \u2022 Dataset [0: MMI | 1: CK+ | 2: CK+48]: ")
    loaders = [MMILoader(), CKPLoader(), CKPLoader(version48=True)]
    loader = loaders[int(choice)]
    loader.load()

    # Training for plotting
    model = DeXpression()
    training_set, validation_test = loader.dataset.slice(portion=0.25)
    stats = model.fit(training_set, validation_test, epochs=1)
    print(stats)

    # Model Cross-Validation
    stats = DeXpression.cross_validate(
        loader.dataset,
        learning_rate=0.001,
        output=True
    )
    cm_list = []

    # Construction of confusion matrix
    for stat in stats:
        matrix = cm(
            stat["actual_labels"],
            stat["predicted_labels"]
        )
        matrix = (matrix.T / matrix.astype(np.float32).sum(axis=1)).T
        labels = [label.name.lower() for label in loader.Labels]
        cm_list.append(pd.DataFrame(matrix, labels, labels))
    averaged_cm = pd.concat(cm_list).groupby(level=0).mean()

    # Saving Confusion Matrix
    ax = sb.heatmap(averaged_cm, annot=True, cmap="YlGnBu")
    ax.set(ylabel="Actual", xlabel="Predicted")
    plt.title("K-Fold Confusion Matrix")
    plt.savefig(
        os.path.join(
            save_path,
            "train",
            f"cm_{['mmi', 'ck+', 'ck+48'][int(choice)]}"
            f"_{time.strftime('%Y%m%d-%H%M%S')}.png"
        ),
        bbox_inches="tight",
        dpi=600
    )
    plt.close()
