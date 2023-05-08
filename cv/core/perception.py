#!/usr/bin/env python
from __future__ import annotations
import os
import cv2
from torch import nn
import numpy as np
from enum import Enum, auto
from frozendict import frozendict
from dataclasses import dataclass
from cv.dataset.loader import CKPLoader, MMILoader


@dataclass
class BoundingBox:

    _tl: tuple[int, int]
    _br: tuple[int, int]
    _width: int
    _height: int

    def draw(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        return cv2.rectangle(frame, self._tl, self._br, color, thickness)

    @property
    def tl(self) -> tuple[int, int]:
        return self._tl

    @property
    def br(self) -> tuple[int, int]:
        return self._br

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


class FaceDetector:

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        max_size: tuple[int, int] = None,
        flags: int = cv2.CASCADE_SCALE_IMAGE,
    ):
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._max_size = max_size
        self._flags = flags
        self._face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            maxSize=self._max_size,
            flags=self._flags
        )
        face_bounding_boxes = []
        if len(faces) > 0:
            face_bounding_boxes = [
                BoundingBox((x, y), (x + w, y + h), w, h)
                for (x, y, w, h) in faces
            ]
        return face_bounding_boxes

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @property
    def min_neighbors(self) -> int:
        return self._min_neighbors

    @property
    def min_size(self) -> tuple[int, int]:
        return self._min_size

    @property
    def max_size(self) -> tuple[int, int]:
        return self._max_size

    @property
    def flags(self) -> int:
        return self._flags


class DeXpression:

    class MType(Enum):
        CKP = len(CKPLoader.Labels)
        MMI = len(MMILoader.Labels)

    def __init__(self, mtype: str = "CKP"):
        if mtype.upper() not in DeXpression.MType.__members__:
            raise ValueError(
                f"mtype must be one of {list(DeXpression.MType.__members__.keys())}"
                f" - provided {mtype}"
            )
        self._cnn = frozendict({
            # PPB
            "conv-1": nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            "pool-1": nn.MaxPool2d(kernel_size=3, stride=2),
            "lnrm-1": nn.LayerNorm([64, 55, 55]),
            # Feat-Ex-1
            "conv-2a": nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1),
            "conv-2b": nn.Conv2d(in_channels=96, out_channels=208, kernel_size=3, padding=1),
            "pool-2a": nn.MaxPool2d(kernel_size=3, padding=1),
            "conv-2c": nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            "pool-2b": nn.MaxPool2d(kernel_size=3, stride=2),
            # Feat-Ex-2
            "conv-3a": nn.Conv2d(in_channels=272, out_channels=96, kernel_size=1),
            "conv-3b": nn.Conv2d(in_channels=96, out_channels=208, kernel_size=3, padding=1),
            "pool-3a": nn.MaxPool2d(kernel_size=3, padding=1),
            "conv-3c": nn.Conv2d(in_channels=272, out_channels=64, kernel_size=1),
            "pool-3b": nn.MaxPool2d(kernel_size=3, stride=2),
        })
        self._classifier = (
            nn.Linear(
                in_features=272*13**2,
                out_features=getattr(DeXpression.MType.CKP, mtype).value
            ),
            nn.LogSoftmax(dim=1),
            nn.BatchNorm2d(272),
            nn.Dropout(p=0.2)
        )

    @property
    def cnn(self):
        return self._cnn

    @property
    def classifier(self) -> tuple:
        return self._classifier
