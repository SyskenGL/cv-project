#!/usr/bin/env python3
from __future__ import annotations
import os
import cv2
import numpy as np
from enum import Enum, auto


class Dataset:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"data and label sizes do not match "
                f"({data.shape[-1]} != {labels.shape[-1]})"
            )
        self._data = data
        self._labels = labels

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

    def load(self, **kwargs) -> None:
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
            if labels[n] < 1 or labels[n] > len(CKPLoader.Labels):
                raise ValueError(
                    f"labels value must be in (1, {len(CKPLoader.Labels)})"
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

    def load(self, **kwargs) -> None:
        data, labels = [], []
        for folder in os.listdir(self.path):
            folder_path = os.path.join(self.path, folder)
            for filename in os.listdir(folder_path):
                image = cv2.imread(
                    os.path.join(folder_path, filename),
                    cv2.IMREAD_GRAYSCALE
                )
                image = cv2.resize(image, kwargs["shape"])
                image = image[..., np.newaxis]
                data.append(image)
                labels.append(getattr(CKPLoader.Labels, folder.upper()).value)
        self._dataset = Dataset(
            np.array(data),
            self.encode(np.array(labels))
        )


class MMILoader(Loader):

    class Labels(Enum):
        pass

    def __init__(self):
        super().__init__()

    @staticmethod
    def encode(labels: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def decode(labels: np.ndarray) -> np.ndarray:
        pass

    def load(self, **kwargs) -> None:
        pass
