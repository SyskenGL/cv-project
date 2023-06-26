#!/usr/bin/env python3
import os
import sys
import cv2
from uuid import uuid1
from cv.core.perception import FaceDetector


if __name__ == "__main__":

    # Parameter checking
    if len(sys.argv) != 3:
        raise ValueError(f"expected source and destination path")
    if not os.path.exists(sys.argv[1]):
        raise ValueError(f"provided source path {sys.argv[1]} does not exist")
    if not os.path.exists(sys.argv[1]):
        raise ValueError(f"provided destination path {sys.argv[2]} does not exist")

    # Creation of the folders based on emotion label
    labels = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
    folders = [os.path.join(sys.argv[2], label) for label in labels]
    for folder in [folder for folder in folders if not os.path.exists(folder)]:
        os.mkdir(folder)

    # Initialize Face Detector
    face_detector = FaceDetector(min_neighbors=7, min_size=(120, 120))
    total_images = [0, 0, 0, 0, 0, 0, 0]

    for emotion, label in enumerate(labels):

        uuid = str(uuid1())[:18]
        images = []
        path = os.path.join(sys.argv[1], label)
        print(f"\u2022 Processing {label} images")

        for filename in os.listdir(path):
            image = cv2.imread(os.path.join(path, filename))
            faces = face_detector.detect(image)
            assert len(faces) > 0
            image = image[
                faces[0].tl[1]:faces[0].br[1],
                faces[0].tl[0]:faces[0].br[0]
            ]
            images.append((filename, image))

        # Saving images
        total_images[emotion] += len(images)
        save_path = os.path.join(sys.argv[2], labels[emotion])
        for i in range(len(images)):
            cv2.imwrite(
                os.path.join(save_path, images[i][0]),
                cv2.cvtColor(images[i][1], cv2.COLOR_BGR2GRAY)
            )
        print(
            f"\t{len(images)} images representing {labels[emotion]} "
            f"saved in folder {os.path.join(save_path)}"
        )

    print(
        f"\u2022 Extraction finished with a total of {sum(total_images)} images:\n" +
        "".join(f"\t{labels[i]}: {total_images[i]}\n" for i in range(len(labels)))
    )
