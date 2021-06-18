import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        self.tip_id = [4,8,12,16,20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS joins the landmark points

        return img

    def findPosition(self, img, handNumber=0, draw=True):
        xList = []
        yList = []
        bbox = []
        height, width, _ = img.shape
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                x,y,z = int(landmark.x * width), int(landmark.y * height), landmark.z
                # print(id, x, y)
                xList.append(x)
                yList.append(y)
                self.landmark_list.append([id,x,y])
                if draw:
                    if id==4:
                        cv2.circle(img, (x,y), 10, (255,0,0), cv2.FILLED)
                    # self.mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS joins the landmark points

        # xmin, xmax = min(xList), max(xList)
        # ymin, ymax = min(yList), max(yList)
        # bbox = xmin, ymin, xmax, ymax

        return self.landmark_list, bbox

    def findFingersUp(self):
        fingers = []
        # Thumb
        if self.landmark_list[4][1] < self.landmark_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # fore fingers
        for item in self.tip_id[1:]:
            if self.landmark_list[item][2] < self.landmark_list[item-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    video_capture = cv2.VideoCapture(0)

    prev_time=0
    current_time=0
    detector = handDetector()
    while True:
        _, img = video_capture.read()
        
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)
        if landmark_list:
            print(landmark_list[8])

        current_time = time.time()
        fps = 1/(current_time-prev_time)
        prev_time = current_time

        cv2.putText(img, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()