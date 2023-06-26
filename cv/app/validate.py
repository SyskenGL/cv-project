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

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data")
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

    # Model K-fold Cross-Validation
    fold_stats = DeXpression.cross_validate(
        loader.dataset,
        output=True,
        mtype=["MMI", "CKP", "CKP48"][int(choice)]
    )
    cm_list = []

    # Construction of K-fold Confusion Matrix
    for fold_stat in fold_stats:
        fold_stat = fold_stat["validation"]
        matrix = cm(
            fold_stat["actual_labels"],
            fold_stat["predicted_labels"]
        )
        matrix = (matrix.T / matrix.astype(np.float32).sum(axis=1)).T
        labels = [label.name.lower() for label in loader.Labels]
        cm_list.append(pd.DataFrame(matrix, labels, labels))
    averaged_cm = pd.concat(cm_list).groupby(level=0).mean()

    # Saving K-fold Confusion Matrix
    ax = sb.heatmap(averaged_cm, annot=True, cmap="YlGnBu")
    ax.set(ylabel="Actual", xlabel="Predicted")
    plt.title("K-Fold Confusion Matrix")
    plt.savefig(
        os.path.join(
            save_path,
            "plots",
            f"{['mmi', 'ckp', 'ckp48'][int(choice)]}"
            f"_cm_{time.strftime('%Y%m%d-%H%M%S')}.png"
        ),
        bbox_inches="tight",
        dpi=600
    )
    plt.close()

    # Gather plot data
    folds_plot_data = [{
        "training": {
            "loss": [],
            "accuracy": []
        },
        "validation": {
            "loss": [],
            "accuracy": []
        }
    } for i in range(len(fold_stats))]

    for i in range(len(fold_stats)):
        for dataset in ["training", "validation"]:
            for metric in ["loss", "accuracy"]:
                folds_plot_data[i][dataset][metric] = [
                    fold_stats[i]["fit"][epoch][dataset][metric]
                    for epoch in range(len(fold_stats[i]["fit"]))
                ]

    # Plotting
    for dataset in ["training", "validation"]:
        for metric in ["loss", "accuracy"]:
            figure, axis = plt.subplots()
            axis.set_xlabel("Epochs", labelpad=10)
            axis.set_ylabel(metric.capitalize(), labelpad=10)
            for i in range(len(folds_plot_data)):
                axis.plot(
                    range(1, len(folds_plot_data[i][dataset][metric])+1),
                    folds_plot_data[i][dataset][metric],
                    label=f"{i}-fold"
                )
            axis.legend(loc=("lower right" if metric == "accuracy" else "upper right"))
            figure.savefig(
                os.path.join(
                    save_path,
                    "plots",
                    f"{['mmi', 'ckp', 'ckp48'][int(choice)]}"
                    f"_plt_{dataset}_{metric}_{time.strftime('%Y%m%d-%H%M%S')}.png"
                ),
                bbox_inches="tight",
                dpi=600
            )
