#!/usr/bin/env python
import cv2
from cv.core.perception import FaceDetector


if __name__ == "__main__":

    vcap = cv2.VideoCapture(0)
    fd = FaceDetector(min_neighbors=5, min_size=(100, 100))

    while True:

