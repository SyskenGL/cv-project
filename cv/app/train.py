import logging
import os
import sys
import time
import matplotlib.pyplot as plt
import cv.dataset.loader as loaders
from cv.core.dexpression import DeXpression


if __name__ == "__main__":

    logging.basicConfig(
        format="\n%(asctime)s [%(levelname)-5.5s] \n%(message)s\n",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "logs",
                    f"{time.strftime('%Y%m%d-%H%M%S')}.log"
                ),
                encoding="utf-8"
            )
        ]
    )

    # Loading Dataset
    choices = [loaders.MMI, loaders.CKP, loaders.CKP48, loaders.FKT]
    choice = None
    while choice not in ["0", "1", "2", "3"]:
        choice = input("\n \u2022 Dataset [0: MMI - 1: CKP - 2: CKP48 - 3: FKT]: ")
    dtype = choices[int(choice)]
    loader = loaders.Loader(dtype)
    loader.load()

    # Model Training
    model = DeXpression(dtype)
    training_set, validation_test = loader.dataset.slice(portion=0.25)
    stats = model.fit(training_set, validation_test)
    model.save()

    # Gather plot data
    plot_data = {
        "training": {
            "loss": [],
            "accuracy": []
        },
        "validation": {
            "loss": [],
            "accuracy": []
        }
    }
    for stat in stats:
        for dataset in plot_data.keys():
            for metric in plot_data[dataset].keys():
                plot_data[dataset][metric].append(stat[dataset][metric])

    # Plotting
    for metric in ["loss", "accuracy"]:
        figure, axis = plt.subplots()
        axis.set_title(f"Training & Validation {metric.capitalize()} Plot")
        axis.set_xlabel("Epochs", labelpad=10)
        axis.set_ylabel(metric.capitalize(), labelpad=10)
        axis.plot(
            range(1, len(plot_data["training"][metric])+1),
            plot_data["training"][metric],
            label="Training"
        )
        axis.plot(
            range(1, len(plot_data["validation"][metric])+1),
            plot_data["validation"][metric],
            label="Validation"
        )
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        axis.set_aspect(abs((x_max - x_min) / (y_max - y_min)) * 0.8, adjustable="box")
        axis.legend(loc=("lower right" if metric == "accuracy" else "upper right"))
        figure.savefig(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "plots",
                f"{dtype.name()}_"
                f"plt_{metric}_{time.strftime('%Y%m%d-%H%M%S')}.png"
            ),
            bbox_inches="tight",
            dpi=600
        )
