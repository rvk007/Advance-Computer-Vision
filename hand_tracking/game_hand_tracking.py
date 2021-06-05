import cv2
import time
import hand_tracking_module as ht


video_capture = cv2.VideoCapture(0)
prev_time=0
current_time=0
detector = ht.handDetector()

while True:
    _, img = video_capture.read()
    
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)
    # if landmark_list:
    #     print(landmark_list[8])

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)