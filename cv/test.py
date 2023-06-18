import logging
import os
import numpy as np
import pandas as pd
import seaborn as sb
from uuid import uuid1
import matplotlib.pyplot as plt
from cv.core.dexpression import DeXpression
from cv.dataset.loader import CKPLoader, MMILoader
from sklearn.metrics import confusion_matrix as cm


if __name__ == "__main__":

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/results")
    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)

    # Loading Dataset
    choice = None
    while choice not in ["0", "1", "2"]:
        choice = input("\n \u2022 Dataset [0: MMI | 1: CK+ | 2: CK+48]: ")
    loaders = [MMILoader(), CKPLoader(), CKPLoader(version48=True)]
    loader = loaders[int(choice)]
    loader.load()

    # Model Cross-Validation
    model = DeXpression()
    stats = model.cross_validate(loader.dataset, output=True)
    cm_list = []

    # Construction of confusion matrix
    for fold_stats in stats:
        validation_stats = fold_stats["validation"]
        matrix = cm(
            validation_stats["actual_labels"],
            validation_stats["predicted_labels"]
        )
        matrix = (matrix.T / matrix.astype(np.float32).sum(axis=1)).T
        labels = [label.name.lower() for label in loader.Labels]
        cm_list.append(pd.DataFrame(matrix, labels, labels))
    averaged_cm = pd.concat(cm_list).groupby(level=0).mean()

    # Saving Confusion Matrix
    ax = sb.heatmap(averaged_cm, annot=True, cmap="YlGnBu")
    ax.set(ylabel="Actual", xlabel="Predicted")
    plt.title("K-Fold CM")
    plt.savefig(
        os.path.join(save_path, f"cm_{str(uuid1())[:18]}.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()
