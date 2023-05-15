#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import cv2
import numpy as np
from enum import Enum, auto



class Dataset:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"data and label sizes do not match "
                f"({data.shape[0]} != {labels.shape[0]})"
            )
        self._data = data
        self._labels = labels

    def slice(
        self,
        portion: float = 0.25,
        shuffle: bool = False
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
        return Dataset(lt_data, lt_labels), Dataset(rt_data, rt_labels)

    def kfold(
        self,
        splits: int = 2,
        shuffle: bool = False
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
                    np.concatenate(data[:k] + data[k+1:], axis=0),
                    np.concatenate(labels[:k] + labels[k+1:], axis=0)
                ),
                Dataset(data[k], labels[k])
            ) for k in range(splits)
        ]

    def shuffle(self, size: int):
        if size > self.size:
            raise ValueError(
                f"largest possible size {self.size}"
                f" - provided {size}"
            )
        choices = np.random.choice(np.random.permutation(size))
        data = self.data[choices, :]
        labels = self.labels[choices, :]
        return Dataset(data, labels)

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

    def __str__(self) -> str:
        return f"[ Data: {str(self.data.shape)} - Labels: {str(self.labels.shape)} ]"


class Loader:

    def __init__(self):
        self._dataset = None
        self._path = None

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def load(self, **kwargs: dict) -> None:
        raise NotImplementedError

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def path(self):
        return self._path


class CKPLoader(Loader):

    class Labels(Enum):
        ANGER = auto()
        CONTEMPT = auto()
        DISGUST = auto()
        FEAR = auto()
        HAPPINESS = auto()
        NEUTRAL = auto()
        SADNESS = auto()
        SURPRISE = auto()

    def __init__(self):
        super().__init__()
        self._dataset = None
        self._path = os.path.join(os.path.dirname(__file__), "raw", "CK+")

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        if len(labels.shape) != 1:
            raise ValueError("labels shape must be 1-dimensional")
        encoded_labels = np.zeros(shape=(labels.shape[0], len(CKPLoader.Labels)))
        for n in range(labels.shape[0]):
            if not 1 <= labels[n] <= len(CKPLoader.Labels):
                raise ValueError(
                    f"labels value must be in [1, {len(CKPLoader.Labels)})"
                    f" - provided {labels[n]}"
                )
            encoded_labels[n][labels[n]-1] = 1
        return encoded_labels

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        if len(labels.shape) != 2 or labels.shape[1] != len(CKPLoader.Labels):
            raise ValueError(
                f"labels must have shape (?, {len(CKPLoader.Labels)})"
                f" - provided {labels.shape}"
            )
        return np.array([
            np.argmax(labels[n, :]) + 1 for n in range(labels.shape[0])
        ])

    def load(self, **kwargs: dict) -> None:
        data, labels = [], []
        for folder in os.listdir(self.path):
            folder_path = os.path.join(self.path, folder)
            for filename in os.listdir(folder_path):
                image = cv2.imread(
                    os.path.join(folder_path, filename),
                    cv2.IMREAD_GRAYSCALE
                )
                image = cv2.resize(image, kwargs.get("shape", (224, 224)))
                image = image[np.newaxis, ...]
                image = image.astype('float32')
                image = image/255 if kwargs.get("normalize", True) else image
                data.append(image)
                labels.append(getattr(CKPLoader.Labels, folder.upper()).value)
        self._dataset = Dataset(
            np.array(data),
            self.encode(np.array(labels))
        )


class MMILoader(Loader):

    class Labels(Enum):
        ANGER = auto()
        DISGUST = auto()
        FEAR = auto()
        HAPPINESS = auto()
        SADNESS = auto()
        SURPRISE = auto()

    def __init__(self):
        super().__init__()

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        pass

    def load(self, **kwargs: dict) -> None:
        pass


