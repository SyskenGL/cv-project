#!/usr/bin/env python
import cv2
from cv.core.perception import FaceDetector
from cv.core.perception import EmotionRecognizer


if __name__ == "__main__":

    vcap = cv2.VideoCapture(0)
    faceDetector = FaceDetector(min_neighbors=5, min_size=(100, 100))
    emotionRecognizer = EmotionRecognizer("CKP")

    while True:
        _, frame = vcap.read()
        faces = vcap.detect(frame)
        for face in faces:
            image = frame[
                face.tl[1]:face.br[1],
                face.tl[0]:face.br[0]
            ]
            emotion = emotionRecognizer.recognize(image)
            print(emotion)
            face.draw(frame)
        cv2.imshow('DeXpression', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
