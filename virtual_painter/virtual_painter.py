import os
import cv2
import time
import numpy as np
import mediapipe as mp

import hand_tracking_module as htm

detector = htm.handDetector(min_detection_confidence=0.85)
path = 'header'
brush_thickness= 15
eraser_thickness = 100
xp, yp = 0, 0

overlay = []
pathList= os.listdir(path)
pathList.sort()
for header_path in pathList:
    header = os.path.join(path, header_path)
    print()
    image = cv2.imread(header)
    image = cv2.resize(image, (1280, 125))
    overlay.append(image)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1280)
video_capture.set(4, 720)
overlap_head = overlay[0]
color = (0,0,255)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    # Import image
    _, img = video_capture.read()
    img = cv2.flip(img, 1)

    # Find hand landmarks
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img, 0, False)
    
    if landmark_list:
        # print(landmark_list)

        # index finger
        x1, y1 = landmark_list[8] [1:]

        # middle finger
        x2, y2 = landmark_list[12][1:]

        # Check which fingers are up
        fingers = detector.findFingersUp()
        # print(fingers)

        # Selection mode: If selectione mode [two fingers] -> select
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # Checking for the selection
            if y1<125:
                # red
                if 250 < x1 < 450:
                    overlap_head = overlay[0]
                    color = (0,0,255)
                # green
                elif 550 < x1 < 750:
                    overlap_head = overlay[1]
                    color = (0,255,0)
                # yellow
                elif 850 < x1 < 1000:
                    overlap_head = overlay[2]
                    color = (0,255,255)
                # eraser-black
                elif 1050 < x1 < 1200:
                    overlap_head = overlay[3]
                    color = (0,0,0)
            cv2.rectangle(img, (x1, y1-30), (x2, y2+30), color, cv2.FILLED)
             

        # Drawing mode: If drawing mode [index finger] -> draw
        if fingers[1] and fingers[2]==0:
            cv2.circle(img, (x1,y1), 15, color, cv2.FILLED)
            if xp==yp and xp==0:
                xp, yp = x1, y1

            if color == (0,0,0):
                cv2.line(img, (xp, yp), (x1,y1), color, eraser_thickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1,y1), color, brush_thickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), color, brush_thickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # Set the header
    img[0:125, 0:1280] = overlap_head
    # one way to show the screen, not ideal
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("ImgCanvas", imgCanvas)
    cv2.waitKey(1)

