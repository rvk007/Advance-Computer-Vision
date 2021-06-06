import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

video_capture = cv2.VideoCapture(0)
prev_time= 0

while True:
    _, img = video_capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, _ = img.shape
            x,y = int(landmark.x * width), int(landmark.y * height)
            # print(id, landmark)
            if id==0:   # point to nose
                cv2.circle(img, (x,y), 10, (255,0,0), cv2.FILLED)

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time=current_time

    cv2.putText(img, str(f'FPS:{int(fps)}'), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)