import logging
import os
import sys
import time
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from cv.core.dexpression import DeXpression
from cv.dataset.loader import CKPLoader, MMILoader


if __name__ == "__main__":

    save_path = os.path.join(Path(os.path.dirname(__file__)).parent, "data")
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

    # Model Training
    model = DeXpression()
    training_set, validation_test = loader.dataset.slice(portion=0.25)
    stats = model.fit(training_set, validation_test)

    # Saving trained model
    torch.save(
        model.state_dict(),
        os.path.join(
            save_path,
            "models",
            f"{['mmi', 'ckp', 'ckp48'][int(choice)]}"
            f"_{time.strftime('%Y%m%d-%H%M%S')}.net"
        )
    )

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
                save_path,
                "plots",
                f"{['mmi', 'ckp', 'ckp48'][int(choice)]}"
                f"_plt_{metric}_{time.strftime('%Y%m%d-%H%M%S')}.png"
            ),
            bbox_inches="tight",
            dpi=600
        )
