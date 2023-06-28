#!/usr/bin/env python3
from __future__ import annotations
from typing import Type
import os
import cv2
import torch
import logging
import numpy as np
from enum import Enum, auto


class Dataset:

    def __init__(self, dtype: Type[DType], data: np.ndarray, labels: np.ndarray):
        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"data and label sizes do not match "
                f"({data.shape[0]} != {labels.shape[0]})"
            )
        if len(labels.shape) != 2 or labels.shape[1] != len(dtype.Labels):
            raise ValueError(
                f"expected labels of shape ({labels.shape[0]}, {len(dtype.Labels)})"
                f" provided {labels.shape}"
            )
        self._dtype = dtype
        self._data = data
        self._labels = labels

    def slice(
        self,
        portion: float = 0.25,
        shuffle: bool = True
    ) -> tuple[Dataset, Dataset]:
        if not 0 < portion <= self.size:
            raise ValueError(
                f"portion must be in (0, {self.size}]"
                f" - provided {portion}"
            )
        choices = (
            np.random.permutation(self.size)
            if shuffle else np.arange(0, self.size)
        )
        rt_dataset_size = int(self.size * portion) if (0 < portion < 1) else portion
        lt_dataset_size = self.size - rt_dataset_size
        rt_data = self.data[choices[lt_dataset_size:], :]
        rt_labels = self.labels[choices[lt_dataset_size:], :]
        lt_data = self.data[choices[:lt_dataset_size], :]
        lt_labels = self.labels[choices[:lt_dataset_size], :]
        return (
            Dataset(self._dtype, lt_data, lt_labels),
            Dataset(self._dtype, rt_data, rt_labels)
        )

    def kfold(
        self,
        splits: int = 5,
        shuffle: bool = True
    ) -> list[tuple[Dataset, Dataset]]:
        if not 1 < splits <= self.data.shape[0]:
            raise ValueError(
                f"splits must be in [2, {self.data.shape[0]}]"
                f" - provided {splits}"
            )
        choices = (
            np.random.permutation(self.size)
            if shuffle else np.arange(0, self.size)
        )
        data = np.array_split(self.data[choices, :], splits, axis=0)
        labels = np.array_split(self.labels[choices, :], splits, axis=0)
        return [
            (
                Dataset(
                    self._dtype,
                    np.concatenate(data[:k] + data[k+1:], axis=0),
                    np.concatenate(labels[:k] + labels[k+1:], axis=0)
                ),
                Dataset(self._dtype, data[k], labels[k])
            ) for k in range(splits)
        ]

    def random(self, size: int) -> Dataset:
        if size > self.size:
            raise ValueError(
                f"largest possible size {self.size}"
                f" - provided {size}"
            )
        choices = np.random.choice(self.size, size)
        data = self.data[choices]
        labels = self.labels[choices]
        return Dataset(self._dtype, data, labels)

    def torch(self):
        data = torch.from_numpy(self._data).type(torch.FloatTensor)
        labels = torch.from_numpy(self._labels).type(torch.FloatTensor)
        return {"data": data, "labels": labels}

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def size(self) -> int:
        return self._labels.shape[0]

    @property
    def shape(self) -> tuple:
        return self._labels.shape

    @property
    def dtype(self) -> Type[DType]:
        return self._dtype

    def __str__(self) -> str:
        return f"[ Data: {str(self.data.shape)} " \
               f"- Labels: {str(self.labels.shape)} ]"


class DType:

    class Labels(Enum):
        pass

    def __init__(self):
        raise Exception(
            f"{self.__class__.__name__}"
            f" is an abstract class and cannot be instantiated"
        )

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def description() -> str:
        raise NotImplementedError

    @staticmethod
    def base_path() -> str:
        return os.path.join(os.path.dirname(__file__), "raw")

    @staticmethod
    def path() -> str:
        raise NotImplementedError


class CKP(DType):

    class Labels(Enum):
        ANGER = 0
        CONTEMPT = auto()
        DISGUST = auto()
        FEAR = auto()
        HAPPINESS = auto()
        SADNESS = auto()
        SURPRISE = auto()

    @staticmethod
    def name() -> str:
        return "ckp"

    @staticmethod
    def description() -> str:
        raise "CKP: Cohn-Kanade Plus"

    @staticmethod
    def path() -> str:
        return os.path.join(DType.base_path(), "ckp")


class CKP48(DType):

    class Labels(Enum):
        ANGER = 0
        CONTEMPT = auto()
        DISGUST = auto()
        FEAR = auto()
        HAPPINESS = auto()
        SADNESS = auto()
        SURPRISE = auto()

    @staticmethod
    def name() -> str:
        return "ckp48"

    @staticmethod
    def description() -> str:
        return "CKP48: Cohn-Kanade Plus 48x48"

    @staticmethod
    def path() -> str:
        return os.path.join(DType.base_path(), "ckp48")


class MMI(DType):

    class Labels(Enum):
        ANGER = 0
        DISGUST = auto()
        FEAR = auto()
        HAPPINESS = auto()
        SADNESS = auto()
        SURPRISE = auto()

    @staticmethod
    def name() -> str:
        return "mmi"

    @staticmethod
    def description() -> str:
        return "MMI: Facial Expression"

    @staticmethod
    def path() -> str:
        return os.path.join(DType.base_path(), "mmi")


class FKT(DType):

    class Labels(Enum):
        ANGER = 0
        DISGUST = auto()
        FEAR = auto()
        HAPPINESS = auto()
        SADNESS = auto()
        SURPRISE = auto()

    @staticmethod
    def name() -> str:
        return "fkt"

    @staticmethod
    def description() -> str:
        return "FKT: Merged MMI & CKP"

    @staticmethod
    def path() -> str:
        return os.path.join(DType.base_path(), "fkt")


class Loader:

    def __init__(self, dtype: Type[DType]):
        self._dataset = None
        self._dtype = dtype
        self._logger = logging.getLogger(self.__class__.__name__)

    def encode(self, labels: np.ndarray) -> np.ndarray:
        if len(labels.shape) != 1:
            raise ValueError("labels shape must be 1-dimensional")
        encoded_labels = np.zeros(shape=(labels.shape[0], len(self._dtype.Labels)))
        for n in range(labels.shape[0]):
            if not 0 <= labels[n] <= len(self._dtype.Labels)-1:
                raise ValueError(
                    f"labels value must be in [0, {len(self._dtype.Labels)-1})"
                    f" - provided {labels[n]}"
                )
            encoded_labels[n][labels[n]] = 1
        return encoded_labels

    def decode(self, labels: np.ndarray, verbose: bool = True) -> np.ndarray:
        if len(labels.shape) != 2 or labels.shape[1] != len(self._dtype.Labels):
            raise ValueError(
                f"labels must have shape (?, {len(self._dtype.Labels)})"
                f" - provided {labels.shape}"
            )
        labels = np.array([
            np.argmax(labels[n, :]) for n in range(labels.shape[0])
        ])
        labels = (
            [self._dtype.Labels(label).name.lower() for label in labels]
            if verbose else labels
        )
        return labels

    def load(self, **kwargs: dict) -> None:
        data, labels = [], []
        for folder in os.listdir(self._dtype.path()):
            folder_path = os.path.join(self._dtype.path(), folder)
            files = os.listdir(folder_path)
            self._logger.debug(f" \u2022 Loading folder {folder} (files: {len(files)})")
            for filename in files:
                image = (
                    cv2.imread(
                        os.path.join(folder_path, filename),
                        cv2.IMREAD_GRAYSCALE
                    ) if kwargs.get("grayscale", True) else
                    cv2.imread(os.path.join(folder_path, filename))
                )
                image = cv2.resize(image, kwargs.get("shape", (224, 224)))
                image = image[np.newaxis, ...]
                image = image.astype('float32')
                image = image/255 if kwargs.get("normalize", True) else image
                data.append(image)
                labels.append(getattr(self._dtype.Labels, folder.upper()).value)
        self._dataset = Dataset(
            self._dtype,
            np.array(data),
            self.encode(np.array(labels))
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset
