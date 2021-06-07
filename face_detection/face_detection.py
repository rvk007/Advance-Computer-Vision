import cv2
import mediapipe as mp
import time

video_capture = cv2.VideoCapture(0)
prev_time=0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    _, img = video_capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # mpDraw.draw_detection(img, detection)
            height, width, _ = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                    int(bboxC.width * width), int(bboxC.height * height)

            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, 
            str(f'{int(detection.score[0]*100)}%'), 
            (bbox[0], bbox[1]-20), 
            cv2.FONT_HERSHEY_PLAIN, 
            1.5, 
            (255,0,255), 
            2)


    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, str(f'FPS:{int(fps)}'), (20,70), cv2.FONT_HERSHEY_PLAIN, 3 ,(0,255,0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)