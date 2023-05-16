#!/usr/bin/env python3
import os
import sys
import cv2
import glob
import numpy as np
from bs4 import BeautifulSoup
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
    labels = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]
    folders = [os.path.join(sys.argv[2], label) for label in labels]
    for folder in [folder for folder in folders if not os.path.exists(folder)]:
        os.mkdir(folder)

    # Initialize Face Detector
    face_detector = FaceDetector(min_neighbors=7, min_size=(120, 120))
    total_images = [0, 0, 0, 0, 0, 0]

    # Loop on all session folders
    for session in os.listdir(sys.argv[1]):

        path = os.path.join(sys.argv[1], session)
        sid = os.path.basename(glob.glob(os.path.join(path, "*.avi"))[0]).split(".")[0]
        print(f"• Processing video {sid}")

        # Get video type
        session_meta = BeautifulSoup(open(os.path.join(path, "session.xml")), "xml")
        view_type = session_meta.find("track").get("view")
        if view_type not in ["0", "2"]:
            print(
                f"\tVideo {sid} discarded due to "
                f"incompatible view type {view_type}"
            )
            continue

        # Get emotion type
        sid_meta = BeautifulSoup(open(os.path.join(path, f"{sid}.xml")), "xml")
        emotion = int(sid_meta.find("Metatag", {"Name": "Emotion"}).get("Value")) - 1
        if emotion > len(labels):
            print(
                f"\tVideo {sid} discarded due to "
                f"incompatible emotion {emotion}"
            )
            continue

        # Get frames
        cap = cv2.VideoCapture(os.path.join(path, f"{sid}.avi"))
        frames, processed = [], []
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(grayscale, (11, 11), 0.0)
            processed.append(gaussian)
        cap.release()

        # Compute iframes
        diff = [
            cv2.sumElems(cv2.absdiff(processed[i], processed[i + 1]))[0]
            for i in range(len(processed) - 1)
        ]
        threshold = np.mean(diff) + 0.65 * np.std(diff)
        iframes = [frames[i] for i in range(len(diff)) if diff[i] >= threshold]
        iframes = [] if len(iframes) <= 2 else iframes[2:]
        if len(iframes) == 0:
            print(f"\tNo iframes extracted for video {sid}")
            continue
        print(f"\t{len(iframes)} iframes extracted for video {sid}")

        # Process the iframes
        if view_type == "0":
            if "oao_aucs.xml" in session_meta.find_all("annotation")[-1].get("filename"):
                print(f"\tCropping iframes")
                iframes = [iframe[0:iframe.shape[0], 100:iframe.shape[1]-100] for iframe in iframes]
            else:
                print(f"\tRotating iframes")
                iframes = [cv2.rotate(iframe, cv2.ROTATE_90_CLOCKWISE) for iframe in iframes]
                iframes = [iframe[10:iframe.shape[0], 0:iframe.shape[1]] for iframe in iframes]
        elif view_type == "2":
            print(f"\tDetecting and cropping face")
            for i in range(len(iframes)):
                faces = face_detector.detect(iframes[i])
                height, width, _ = iframes[i].shape
                assert len(faces) > 0
                iframes[i] = (
                    iframes[i][100:height, int(width/2):width]
                    if faces[0].br[0] > (width/2) else
                    iframes[i][100:height, 10:int(width/2)]
                )

        # Saving images
        total_images[emotion] += len(iframes)
        save_path = os.path.join(sys.argv[2], labels[emotion])
        for i in range(len(iframes)):
            cv2.imwrite(os.path.join(save_path, f"{sid}-{i:03}.png"), iframes[i])
        print(
            f"\t{len(iframes)} images representing {labels[emotion]} "
            f"saved in folder {os.path.join(save_path)}"
        )

    print(
        f"• Extraction finished with a total of {sum(total_images)} images:\n" +
        "".join(f"\t{labels[i]}: {total_images[i]}\n" for i in range(len(labels)))
    )
