#!/usr/bin/env python
from __future__ import annotations
from typing import Type
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from cv.dataset.loader import DType
from cv.core.dexpression import DeXpression


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

    @property
    def area(self) -> int:
        return self._height * self._width


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
        self._face_cascade = cv2.CascadeClassifier(os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml"
        ))

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
            face_bounding_boxes.sort(key=lambda bb: bb.area, reverse=True)
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


class EmotionRecognizer:

    def __init__(self, dtype: Type[DType]):
        self._path = os.path.join(Path(os.path.dirname(__file__)), "models")
        self._model = DeXpression(dtype)
        self._model.load()

    def recognize(
        self,
        frame: np.ndarray,
        normalize: bool = True,
        threshold: float = 0.5
    ) -> str:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, (224, 224))
        image = resized[np.newaxis, np.newaxis, ...]
        image = image.astype('float32')
        image = image / 255 if normalize else image
        tensor = torch.from_numpy(image)
        return self._model.predict(tensor, threshold)[0]

    @property
    def path(self) -> str:
        return self._path

    @property
    def model(self) -> DeXpression:
        return self._model
