#!/usr/bin/env python
import cv2
import cv.dataset.loader as loaders
from cv.core.perception import FaceDetector
from cv.core.perception import EmotionRecognizer


if __name__ == "__main__":

    choices = [loaders.MMI, loaders.CKP, loaders.CKP48, loaders.FKT]
    choice = None
    while choice not in ["0", "1", "2", "3"]:
        choice = input("\n \u2022 Dataset [0: MMI - 1: CKP - 2: CKP48 - 3: FKT]: ")

    vcap = cv2.VideoCapture(0)
    faceDetector = FaceDetector(min_neighbors=5, min_size=(100, 100))
    dtype = choices[int(choice)]
    emotionRecognizer = EmotionRecognizer(dtype)

    emotionRecognizerDelay = 60
    counter = 0
    emotion = ""

    while True:
        ret, frame = vcap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        counter -= 1
        faces = faceDetector.detect(frame)
        for face in faces:
            image = frame[
                face.tl[1]:face.br[1],
                face.tl[0]:face.br[0]
            ]
            if counter <= 0:
                counter = emotionRecognizerDelay
                emotion = emotionRecognizer.recognize(image)
                emotion = emotion if emotion else "unknown"
            cv2.putText(
                frame,
                emotion.capitalize(),
                (face.tl[0]+15, face.br[1]-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color=(158, 199, 44), thickness=2
            )
            face.draw(frame, color=(158, 199, 44), thickness=2)
        cv2.imshow('DeXpression', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
