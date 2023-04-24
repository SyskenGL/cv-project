#!/usr/bin/env python
import cv2
import logging
import numpy as np


class FaceDetector:

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple = (30, 30),
        max_size: tuple = None,
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

    def detect(self, frame: np.ndarray) -> list[dict]:
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
            for (x, y, w, h) in faces:
                face_bounding_boxes.append({
                    "tl": (x, y),
                    "br": (x + w, y + h),
                    "w": w,
                    "h": h
                })
        return face_bounding_boxes

    @property
    def scale_factor(self) -> float:
        return self.__scale_factor

    @property
    def min_neighbors(self) -> int:
        return self.__min_neighbors

    @property
    def min_size(self) -> tuple:
        return self.__min_size

    @property
    def max_size(self) -> tuple:
        return self.__max_size

    @property
    def flags(self) -> int:
        return self.__flags
