from __future__ import annotations
import logging
import math
import torch
import torch.nn as nn
from uuid import uuid1
from enum import Enum
from typing import Optional
from cv.dataset.loader import Dataset
from cv.dataset.loader import CKPLoader, MMILoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeXpression(nn.Module):

    class MType(Enum):
        CKP = len(CKPLoader.Labels)
        MMI = len(MMILoader.Labels)

    def __init__(self, mtype: str = "CKP"):
        super().__init__()
        if mtype.upper() not in DeXpression.MType.__members__:
            raise ValueError(
                f"mtype must be one of "
                f"{list(DeXpression.MType.__members__.keys())}"
                f" - provided {mtype}"
            )
        out_features = getattr(DeXpression.MType.CKP, mtype).value
        # PPB
        self._conv_1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7,
            stride=2, padding=3
        )
        self._pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self._lnrm_1 = nn.LayerNorm([64, 55, 55])
        # Feat-Ex-1
        self._conv_2a = nn.Conv2d(
            in_channels=64, out_channels=96,
            kernel_size=1,
            stride=1, padding=0
        )
        self._conv_2b = nn.Conv2d(
            in_channels=96, out_channels=208,
            kernel_size=3,
            stride=1, padding=1
        )
        self._pool_2a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self._conv_2c = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=1,
            stride=1, padding=0
        )
        self._pool_2b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # Feat-Ex-2
        self._conv_3a = nn.Conv2d(
            in_channels=272, out_channels=96,
            kernel_size=1,
            stride=1, padding=0
        )
        self._conv_3b = nn.Conv2d(
            in_channels=96, out_channels=208,
            kernel_size=3,
            stride=1, padding=1
        )
        self._pool_3a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self._conv_3c = nn.Conv2d(
            in_channels=272, out_channels=64,
            kernel_size=1,
            stride=1, padding=0
        )
        self._pool_3b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # Classifier
        self._lfc = nn.Linear(in_features=272 * 13 ** 2, out_features=out_features)
        self._bnorm = nn.BatchNorm2d(272)
        self._dropout = nn.Dropout(p=0.2)
        self._softmax = nn.LogSoftmax(dim=1)
        self._loss = nn.NLLLoss()
        self._logger = logging.getLogger(
            f"{self.__class__.__name__}{str(uuid1())[:18]}"
        )

    def forward(
        self,
        in_data: torch.Tensor,
        dropout: bool = True,
        batch_normalization: bool = True
    ) -> torch.Tensor:
        # PPB
        conv_1_out = nn.functional.relu(self._conv_1(in_data))
        pool_1_out = self._pool_1(conv_1_out)
        lnrm_1_out = self._lnrm_1(pool_1_out)
        # Feat-Ex-1
        conv_2a_out = nn.functional.relu(self._conv_2a(lnrm_1_out))
        conv_2b_out = nn.functional.relu(self._conv_2b(conv_2a_out))
        pool_2a_out = self._pool_2a(lnrm_1_out)
        conv_2c_out = nn.functional.relu(self._conv_2c(pool_2a_out))
        cat_2_out = torch.cat((conv_2b_out, conv_2c_out), 1)
        pool_2b_out = self._pool_2b(cat_2_out)
        # Feat-Ex-2
        conv_3a_out = nn.functional.relu(self._conv_3a(pool_2b_out))
        conv_3b_out = nn.functional.relu(self._conv_3b(conv_3a_out))
        pool_3a_out = self._pool_3a(pool_2b_out)
        conv_3c_out = nn.functional.relu(self._conv_3c(pool_3a_out))
        cat_3_out = torch.cat((conv_3b_out, conv_3c_out), 1)
        pool_3b_out = self._pool_3b(cat_3_out)
        # Classifier
        pool_3b_out = self._dropout(pool_3b_out) if dropout else pool_3b_out
        pool_3b_out = self._bnorm(pool_3b_out) if batch_normalization else pool_3b_out
        pool_3b_flatten = torch.flatten(pool_3b_out, start_dim=1)
        output = self._lfc(pool_3b_flatten)
        logits = self._softmax(output)
        return logits

    def validate(self, dataset: Dataset) -> dict:
        if dataset.size == 0:
            raise ValueError("provided an empty validation set.")
        self.eval()
        with torch.no_grad():
            torch_dataset = dataset.torch()
            predictions = self(torch_dataset["data"])
            _, predicted_labels = torch.max(predictions, 1)
            _, actual_labels = torch.max(torch_dataset["labels"], 1)
            correctness = predicted_labels.eq(actual_labels)
            return {
                "hits": int(torch.count_nonzero(correctness)),
                "loss": self._loss(predictions, actual_labels).item(),
                "accuracy": torch.mean(correctness.type(torch.FloatTensor)).item()
            }

    def fit(
        self,
        training_set: Dataset,
        validation_set: Optional[Dataset] = None,
        batch_size: int = 32,
        epochs: int = 25,
        learning_rate: float = 0.001
    ) -> list:

        if training_set.size == 0:
            raise ValueError("provided an empty training set.")
        if validation_set and validation_set.size == 0:
            raise ValueError("provided an empty validation set.")
        if not 0 < batch_size <= training_set.size:
            raise ValueError(f"batch_size must be in [1, {training_set.size}].")
        if epochs <= 0:
            raise ValueError(f"epochs must be greater than 0.")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        epoch_stats = []
        torch_training_set = training_set.torch()
        n_batches = math.ceil(training_set.size / batch_size)

        for epoch in range(epochs):

            self.train()

            running_training_accuracy = 0
            running_training_loss = 0
            running_training_hits = 0

            for i in range(0, training_set.size, batch_size):

                optimizer.zero_grad()
                data = torch_training_set["data"][i: i + batch_size].to(DEVICE)
                labels = torch_training_set["labels"][i: i + batch_size].to(DEVICE)
                predictions = self(data)
                _, predicted_labels = torch.max(predictions, 1)
                _, actual_labels = torch.max(labels, 1)
                correctness = predicted_labels.eq(actual_labels)
                loss = self._loss(predictions, actual_labels)
                loss.backward()
                optimizer.step()
                accuracy = torch.mean(correctness.type(torch.FloatTensor))
                running_training_hits += int(torch.count_nonzero(correctness))
                running_training_accuracy += accuracy.item()
                running_training_loss += loss.item()

            epoch_stats.append({
                "training": {
                    "hits": running_training_hits,
                    "loss": running_training_loss / n_batches,
                    "accuracy": running_training_accuracy / n_batches
                },
                "validation": self.validate(validation_set) if validation_set else None
            })

            learning_method = (
                "on-line" if batch_size == 1 else
                ("full-batch" if batch_size == training_set.size else "mini-batch")
            )
            log = f"\033[1m \u25fc Epoch \033[0m{epoch + 1} of {epochs} - [{learning_method}]\n"
            log += f"\n\033[1m   \u2022 Training size:\033[0m {training_set.size}"
            for key in epoch_stats[-1]["training"].keys():
                value = epoch_stats[-1]["training"][key]
                log += f"\n\033[1m   \u2022 Training {key}:\033[0m {round(value, 3)}"
            log += "\n"
            if validation_set:
                log += f"\n\033[1m   \u2022 Validation size:\033[0m {validation_set.size}"
                for key in epoch_stats[-1]["validation"].keys():
                    value = epoch_stats[-1]["validation"][key]
                    log += f"\n\033[1m   \u2022 Validation {key}:\033[0m {round(value, 3)}"
                log += "\n"
            self._logger.info(log)

        return epoch_stats



logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)

loader = CKPLoader()
loader.load()

model = DeXpression()



print(model.fit(*loader.dataset.slice(0.2, True)))