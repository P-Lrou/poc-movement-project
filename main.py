import cv2
import PoseModule
import mediapipe as mp
import numpy as np

coords = []


def main():
    compter = 0
    limitCompter = 10000
    selectedPoint = 19

    cap = cv2.VideoCapture(0)
    detector = PoseModule.poseDetector()
    while True:
        compter += 1
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1920, 1080))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) > 0:
            length = lmList[19][2]
            per = np.interp(int(length), [157, 285], [100, 0])
            poseBar = np.interp(length, [157, 285], [150, 400])
            cv2.circle(img, (lmList[selectedPoint][1], lmList[selectedPoint][2]),
                       10, (255, 0, 0), cv2.FILLED)
            if compter % 1 == 0:
                tempCoord = lmList[selectedPoint]
                tempCoord.pop(0)
                coords.append(tempCoord)
            if compter > limitCompter:
                compter = 0

            coords2 = np.array(coords)
            coords2 = coords2.reshape((-1, 1, 2))
            cv2.polylines(img, [coords2], False, (255, 0, 0), 10)

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
