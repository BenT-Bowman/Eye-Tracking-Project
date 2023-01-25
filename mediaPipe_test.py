import cv2
import numpy as np
import mediapipe as mp
import math
import random
import time
import pandas as pd
#import pandas
# init part find the absolute path for these two .xml files (crucial)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class eye_detection:
    def __init__(self):
        self.white_dimensions = [500, 500]
        self.frame = None
        self.white = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.gaze_calibration = []
        self.temp_calibration_color = []

        self.time_tracking = time.time()
        self.temp_calibration_color_vote = []
        self.landmark_df = None

    def pandas_column_maker(self, length):
        arr = []
        for i in range(0,length):
            arr.append(str(i)+"x")
            arr.append(str(i)+"y")
        return arr
    
    #Right side, 264
    # Left side of face 127
    # Bridge of nose 122
    # Top of chin 18
    # middle of forhead 151
    def landmark_calculations(self, frame_window, i):
        print(i)
        frame_h, frame_w, _ = frame_window.shape
        output = self.face_mesh.process(frame_window)
        landmark_points = output.multi_face_landmarks
        if landmark_points:
            landmarks = landmark_points[0].landmark
            landmarks = np.array(landmarks)
            series_for_database = np.array([])
            landmark_points_list = []
            for landmark in landmarks[468:473]: # [474:478] #[468:473]
                series_for_database = np.append(series_for_database, [(landmark.x-landmarks[464].x), (landmark.y-landmarks[464].y)])
                cv2.circle(self.white, (int((landmark.x-landmarks[464].x)*self.white_dimensions[0]+self.white_dimensions[0]/2), int((landmark.y-landmarks[464].y)*self.white_dimensions[1]+self.white_dimensions[1]/2)), 3, (255,0,0))
            # 473:478
            #print(series_for_database)
            for landmark in landmarks[473:478]: # [474:478] #[468:473]
                series_for_database = np.append(series_for_database, [(landmark.x-landmarks[464].x), (landmark.y-landmarks[464].y)])
                cv2.circle(self.white, (int((landmark.x-landmarks[464].x)*self.white_dimensions[0]+self.white_dimensions[0]/2), int((landmark.y-landmarks[464].y)*self.white_dimensions[1]+self.white_dimensions[1]/2)), 3, (255,0,0))
            for landmark in landmarks[325:350]: # [474:478] #[468:473]
                series_for_database = np.append(series_for_database, [(landmark.x-landmarks[464].x), (landmark.y-landmarks[464].y)])
                cv2.circle(self.white, (int((landmark.x-landmarks[464].x)*self.white_dimensions[0]+self.white_dimensions[0]/2), int((landmark.y-landmarks[464].y)*self.white_dimensions[1]+self.white_dimensions[1]/2)), 3, (255,0,0))
            for landmark in landmarks[125:150]: # [474:478] #[468:473]
                series_for_database = np.append(series_for_database, [(landmark.x-landmarks[464].x), (landmark.y-landmarks[464].y)])
                cv2.circle(self.white, (int((landmark.x-landmarks[464].x)*self.white_dimensions[0]+self.white_dimensions[0]/2), int((landmark.y-landmarks[464].y)*self.white_dimensions[1]+self.white_dimensions[1]/2)), 3, (255,0,0))
            columns = self.pandas_column_maker(int(len(series_for_database)/2))
            #print(len(series_for_database))
            row = pd.Series(series_for_database).to_frame().T
            row.columns=columns
            self.landmark_df = pd.concat([self.landmark_df, row], ignore_index=True)
            """if self.landmark_df is None:
                self.landmark_df = pd.DataFrame()
                self.landmark_df = pd.concat([self.landmark_df, row])
            else:
                pass"""
            xl = int(landmarks[i].x * frame_w)
            yl = int(landmarks[i].y * frame_h)
            # xl = int(landmarks[380].x * frame_w)
            # yl = int(landmarks[380].y * frame_h)
            # xl2 = int(landmarks[385].x * frame_w)
            # yl2 = int(landmarks[385].y * frame_h)
            cv2.circle(self.frame, (xl, yl), 3, (0,255,0))
            # cv2.circle(self.frame, (xl2, yl2), 3, (0,255,0))
            # print(landmarks[380].y-landmarks[385].y)
            # cv2.circle(self.white, (int(landmarks[i].x*self.white_dimensions[0]), int(landmarks[i].y*self.white_dimensions[1])),3, (0,255,0) )
        return None
#385 380 for blinking
#362 for inner part of eye

    def main_loop(self):
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()== False): 
            print("Error opening video  file")
            return
        i = 263
        while True:
            ret, self.frame = cap.read()
            frame_h, frame_w, _ = self.frame.shape
            self.white_dimensions = [frame_w, frame_h]
            self.white = np.zeros([self.white_dimensions[1],self.white_dimensions[0],3],dtype=np.uint8)
            self.white.fill(255)
            self.landmark_calculations(self.frame, i)
            cv2.imshow('Video', self.frame)
            cv2.imshow('white', self.white)
            pressedKey = cv2.waitKey(1) & 0xFF

            if pressedKey == ord('q'):
                print(self.landmark_df)
                break
            elif pressedKey == ord('a'):
                i += 1
            elif pressedKey == ord('s'):
                i -= 1
            elif pressedKey == ord('d'):
                i += 10
            elif pressedKey == ord('f'):
                i -= 10
if __name__ == "__main__":
    eyes = eye_detection()
    eyes.main_loop()

