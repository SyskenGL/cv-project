#!/usr/bin/env python
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class BoundingBox:

    __tl: tuple[int, int]
    __br: tuple[int, int]
    __width: int
    __height: int

    def draw(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        return cv2.rectangle(frame, self.__tl, self.__br, color, thickness)

    @property
    def tl(self) -> tuple[int, int]:
        return self.__tl

    @property
    def br(self) -> tuple[int, int]:
        return self.__br

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height


class FaceDetector:

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        max_size: tuple[int, int] = None,
        flags: int = cv2.CASCADE_SCALE_IMAGE,
    ):
        self.__scale_factor = scale_factor
        self.__min_neighbors = min_neighbors
        self.__min_size = min_size
        self.__max_size = max_size
        self.__flags = flags
        self.__face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=self.__scale_factor,
            minNeighbors=self.__min_neighbors,
            minSize=self.__min_size,
            maxSize=self.__max_size,
            flags=self.__flags
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
        return self.__scale_factor

    @property
    def min_neighbors(self) -> int:
        return self.__min_neighbors

    @property
    def min_size(self) -> tuple[int, int]:
        return self.__min_size

    @property
    def max_size(self) -> tuple[int, int]:
        return self.__max_size

    @property
    def flags(self) -> int:
        return self.__flags
