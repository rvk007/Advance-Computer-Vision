import cv2
import mediapipe as mp
import time

prev_time = 0
video_capture = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    _, img =  video_capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # print(results)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: # number of faces
            # mpDraw.draw_landmarks(img, face_landmarks)
            mpDraw.draw_landmarks(
                img, 
                face_landmarks, 
                mpFaceMesh.FACE_CONNECTIONS,
                drawSpec, drawSpec) # joins the points outlining the face, eyebrows, eyes, nose and lips

            for idx, landmark in enumerate(face_landmarks.landmark):  #468 points
                height, width, _ = img.shape
                x,y = int(landmark.x * width), int(landmark.y * height)
                print(idx, x,y)


    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, f'FPS:{int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
