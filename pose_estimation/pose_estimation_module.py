import cv2
import mediapipe as mp
import time

class poseEstimation():
    def __init__(self, static_image_mode=False, upper_body_only=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
            self.static_image_mode,
            self.upper_body_only,
            self.smooth_landmarks,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

    def findPosition(self, img, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = img.shape
                x,y = int(landmark.x * width), int(landmark.y * height)
                # print(id, landmark)
                landmark_list.append([id, x, y])
                if draw:
                    # self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                    if id==0:   # point to nose
                        cv2.circle(img, (x,y), 10, (255,0,0), cv2.FILLED)
        return img, landmark_list

def main():
    video_capture = cv2.VideoCapture(0) # you can give video here as well cv2.VideoCapture('video.mp4')
    
    poseEstimator = poseEstimation()
    prev_time= 0

    while True:
        _, img = video_capture.read()
        poseEstimator.findPose(img)
        img, landmark_list = poseEstimator.findPosition(img)
        # if landmark_list:
        #     print(landmark_list)
        
        current_time = time.time()
        fps = 1/(current_time-prev_time)
        prev_time=current_time

        cv2.putText(img, str(f'FPS:{int(fps)}'), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()