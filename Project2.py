import cv2
import numpy as np
import mediapipe as mp
import math
import random
import time
# init part find the absolute path for these two .xml files (crucial)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class eye_detection:
    def __init__(self):
        self.white_dimensions = 1000
        self.frame = None
        self.white = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        self.gaze_calibration = []
        self.temp_calibration_color = []
        self.calibrating = False
        self.calibration_collection = []
        self.calibration_time = None

        self.head_orientation_list = []

        self.time_tracking = time.time()
        self.temp_calibration_color_vote = []
        self.prev_user_choice = None
        self.last_selecting = time.time()

        self.current_keyboard_index = 0
        self.blink_count = 0

    def landmark_to_pupil_avg(self,data):
        """ Finds the pupil using 4 landmark points passed through data paramater """
        avg_x = 0
        avg_y = 0
        if not data:
            return
        for landmark in data:
            x, y = landmark
            avg_x += x
            avg_y += y
        avg_x = avg_x/4
        avg_y = avg_y/4
        return [avg_x, avg_y]
    #Right side, 264
    # Left side of face 127
    # Bridge of nose 122
    # Top of chin 18
    # middle of forhead 151

    def landmark_calculations(self, frame_window):
        """Using Mediapipe face_mesh landmark calculations find the four points around the iris, it also normalizes
        the landmark x/y to the corners of the eye. Additionally 2 points tracking the eyelids for blinking.  """
        frame_h, frame_w, _ = frame_window.shape
        output = self.face_mesh.process(frame_window)
        landmark_points = output.multi_face_landmarks
        if landmark_points:
            landmarks = landmark_points[0].landmark
            landmark_points_list = []
            # landmark_reference_points are for reducing noise and ensuring looking actually chooses the option not moving the head
            """landmark_reference_point_x = (landmarks[263].x+landmarks[362].x)/2
            landmark_reference_point_y = (landmarks[263].y+landmarks[362].y)/2
            for landmark in landmarks[474:478]:
                landmark_points_list.append((landmark_reference_point_x-landmark.x, landmark_reference_point_y-landmark.y))"""
            return landmark_points_list
        return None
    
    def calc_calibration_point(self, pupil):
        #print(self.calibrating)
        """Over the course of .10 seconds collect and then average the collected calibration points."""
        self.calibration_collection.append(pupil)
        if time.time() - self.calibration_time > 0.1:
            numpy_calibration_collection = np.array(self.calibration_collection)
            #print(numpy_calibration_collection[:,0])
            self.gaze_calibration.append([np.mean(numpy_calibration_collection[:,0]), np.mean(numpy_calibration_collection[:,1])])
            self.temp_calibration_color
            self.temp_calibration_color.append([np.uint8(random.randint(0,255)), 
                                                   np.uint8(random.randint(0,255)), 
                                                   np.uint8(random.randint(0,255))])
            self.temp_calibration_color_vote.append(0)
            self.calibrating = False
            return
    

    def main_loop(self):
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()== False): 
            print("Error opening video  file")
            return

        while True:
            self.white = np.zeros([self.white_dimensions,self.white_dimensions,3],dtype=np.uint8)
            self.white.fill(255)
            ret, self.frame = cap.read()
            height, width, _ = self.frame.shape
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            landmark_points_list = self.landmark_calculations(self.frame)
            discovered_pupil = self.landmark_to_pupil_avg(landmark_points_list)
            
            cv2.imshow('Video', self.frame)
            cv2.imshow('Goofy', self.white)
            pressedKey = cv2.waitKey(1) & 0xFF
            
            if pressedKey == ord('q'):
                break
            elif pressedKey == ord('x'):
                self.gaze_calibration.clear()
            elif pressedKey == ord('s'):
                if discovered_pupil is None:
                    continue
                # self.calibrating = True
                # self.calibration_collection.clear()
                # self.calibration_time = time.time()
                self.gaze_calibration.append(discovered_pupil)
                self.temp_calibration_color.append([np.uint8(random.randint(0,255)), 
                                                   np.uint8(random.randint(0,255)), 
                                                   np.uint8(random.randint(0,255))])
                self.temp_calibration_color_vote.append(0)
                print(self.gaze_calibration)

            if self.calibrating == True:
                self.calc_calibration_point(discovered_pupil)

if __name__ == "__main__":
    eyes = eye_detection()
    eyes.main_loop()

