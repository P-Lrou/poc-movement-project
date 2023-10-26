import cv2
import PoseModule
import mediapipe as mp
import numpy as np
from datetime import datetime

coords = []
start_time = None


def main():
    counter = 0
    limit_counter = 10000
    selected_point = 0
    time_limit = 10

    cap = cv2.VideoCapture(0)
    detector = PoseModule.poseDetector()
    while True:
        res = timer().total_seconds()
        counter += 1
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1920, 1080))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) > 0:
            if counter % 1 == 0 and res < time_limit:
                tempCoord = lmList[selected_point]
                tempCoord.pop(0)
                coords.append(tempCoord)
            if counter > limit_counter:
                counter = 0

            coords2 = np.array(coords)
            coords2 = coords2.reshape((-1, 1, 2))
            if res > time_limit:
                cv2.polylines(img, [coords2], False, (255, 0, 0), 10)
            elif len(lmList) > 0:
                cv2.circle(
                    img, (lmList[selected_point][0], lmList[selected_point][1]), 10, (255, 0, 0), cv2.FILLED)
                cv2.putText(img, str(
                    res), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)


def timer():
    now = datetime.now()
    delta = now - start_time
    return delta


if __name__ == "__main__":
    start_time = datetime.now()
    main()
