import cv2
import mediapipe as mp
import time
   
class faceMeshDetection():
    def __init__(self, max_num_faces=1, thickness=1, circle_radius=2):
        self.max_num_faces = max_num_faces
        self.thickness = thickness
        self.circle_radius = circle_radius

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.max_num_faces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = self.thickness, circle_radius=self.circle_radius)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks: # number of faces
                # mpDraw.draw_landmarks(img, face_landmarks)
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, 
                        face_landmarks, 
                        self.mpFaceMesh.FACE_CONNECTIONS,
                        self.drawSpec, self.drawSpec) # joins the points outlining the face, eyebrows, eyes, nose and lips

                face = []
                for idx, landmark in enumerate(face_landmarks.landmark):  #468 points
                    height, width, _ = img.shape
                    x,y = int(landmark.x * width), int(landmark.y * height)
                    # cv2.putText(img, f'{idx}', (x,y), cv2.FONT_HERSHEY_PLAIN, 0.7 ,(0,255,0),1) # shows index of each face landmark 
                    # if idx==0:  # highlight a particular face landmark with a circle
                    #     cv2.circle(img, (x,y), 5, (255,0,0), cv2.FILLED)
                    # print(idx, x,y)
                    face.append([idx,x,y])
                faces.append(face)

        return img, faces


def main():
    prev_time = 0
    video_capture = cv2.VideoCapture(0)
    

    while True:
        _, img =  video_capture.read()
        faceMeshDetector = faceMeshDetection()
        img, faces = faceMeshDetector.findFaceMesh(img)
        if faces:
            print(len(faces))
        current_time = time.time()
        fps = 1/(current_time-prev_time)
        prev_time = current_time

        cv2.putText(img, f'FPS:{int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()