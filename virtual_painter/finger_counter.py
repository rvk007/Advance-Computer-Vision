# Works with right hand
import os
import time

import cv2
import mediapipe as mp
import hand_tracking_module as htm

video_capture = cv2.VideoCapture(0)
prev_time = 0

folderPath = "Images"
overlayList = [0]*6
for file in os.listdir(folderPath):
    overlayList[int(file.split('.')[0])] = cv2.imread(os.path.join(folderPath, file))

detector = htm.handDetector(max_num_hands=1, min_detection_confidence=0.75)

tip_id = [4,8,12,16,20]

while True:
    _, img = video_capture.read()
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)
    # print(landmark_list)
    finger=0
    if landmark_list:
        fingers = []
        # if landmark_list[8][2] < landmark_list[6][2]:
        #     print("Index finger open")

        # Thumb
        if landmark_list[4][1] > landmark_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # fore fingers
        for item in tip_id[1:]:
            if landmark_list[item][2] < landmark_list[item-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        finger = sum(fingers)

        height, width, _ = overlayList[0].shape
        img[100:height+100, 10:width+10] = overlayList[finger]

        cv2.rectangle(img, (10,460), (150, 600), (0.255,0), cv2.FILLED)
        cv2.putText(img, str(finger), (40,550), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 10)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(img , f'FPS:{int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)