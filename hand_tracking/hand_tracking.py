import cv2
import mediapipe as mp
import time

video_capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

prev_time=0
current_time=0

while True:
    success, img = video_capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                x,y,z = int(landmark.x * width), int(landmark.y * height), landmark.z
                # print(id, x, y)
                if id==4:
                    cv2.circle(img, (x,y), 10, (0,255,0), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS joins the landmark points

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)