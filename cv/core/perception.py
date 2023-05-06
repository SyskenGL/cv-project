#!/usr/bin/env python
import os
import cv2
import numpy as np
from dataclasses import dataclass


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
