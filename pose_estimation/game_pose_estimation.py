import cv2
import time
import pose_estimation_module as pe

video_capture = cv2.VideoCapture(0)
    
poseEstimator = pe.poseEstimation()
prev_time= 0

while True:
    _, img = video_capture.read()
    poseEstimator.findPose(img)
    img = poseEstimator.findPosition(img, draw=False)
    
    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time=current_time

    cv2.putText(img, str(f'FPS:{int(fps)}'), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)