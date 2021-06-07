import cv2
import mediapipe as mp
import time

class faceDetection():
    def __init__(self, min_detection_confidence=0.60):
        self.min_detection_confidence = min_detection_confidence

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                height, width, _ = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                        int(bboxC.width * width), int(bboxC.height * height)
                bboxes.append([id, bbox, detection.score[0]])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, 
                    str(f'{int(detection.score[0]*100)}%'), 
                    (bbox[0], bbox[1]-20), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5, 
                    (255,0,255), 
                    2)

        return img, bboxes
    
    def fancyDraw(self, img, bbox, length=30, thickness=5, rectangle_thickness=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255,0,255), rectangle_thickness)
        # top-left
        cv2.line(img, (x,y), (x+length, y), (255,0,255), thickness)
        cv2.line(img, (x,y), (x, y+length), (255,0,255), thickness)

        # top-right
        cv2.line(img, (x1,y), (x1-length, y), (255,0,255), thickness)
        cv2.line(img, (x1,y), (x1, y+length), (255,0,255), thickness)

        # bottom-left
        cv2.line(img, (x,y1), (x+length, y1), (255,0,255), thickness)
        cv2.line(img, (x,y1), (x, y1-length), (255,0,255), thickness)

        # bottom-right
        cv2.line(img, (x1,y1), (x1-length, y1), (255,0,255), thickness)
        cv2.line(img, (x1,y1), (x1, y1-length), (255,0,255), thickness)

        return img

def main():
    video_capture = cv2.VideoCapture(0)
    prev_time=0

    while True:
        _, img = video_capture.read()
        faceDetector = faceDetection()
        img, bboxes = faceDetector.findFaces(img, draw=True)
        if bboxes:
            print(bboxes)

        current_time = time.time()
        fps = 1/(current_time-prev_time)
        prev_time = current_time

        cv2.putText(img, str(f'FPS:{int(fps)}'), (20,70), cv2.FONT_HERSHEY_PLAIN, 3 ,(0,255,0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()