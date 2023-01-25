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

        self.eye_drawing_calibration = []
        self.avg_eye_position = []
        self.avg_eye_position_time = None

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
    def landmark_calculations(self, frame_window, x, y):
        """Using Mediapipe face_mesh landmark calculations find the four points around the iris, it also normalizes
        the landmark x/y to the corners of the eye. Additionally 2 points tracking the eyelids for blinking.  """
        frame_h, frame_w, _ = frame_window.shape
        output = self.face_mesh.process(frame_window)
        landmark_points = output.multi_face_landmarks
        if landmark_points:
            landmarks = landmark_points[0].landmark
            landmark_points_list = []
            # landmark_reference_points are for reducing noise and ensuring looking actually chooses the option not moving the head
            landmark_reference_point_x = (landmarks[263].x+landmarks[362].x)/2
            landmark_reference_point_y = (landmarks[263].y+landmarks[362].y)/2
            for landmark in landmarks[474:478]:
                landmark_points_list.append((landmark_reference_point_x-landmark.x, landmark_reference_point_y-landmark.y))
            # tracking blinking.  0.005 is hardcoded so might not work for different cameras... This'll be something I need to fix.
            if landmarks[380].y-landmarks[385].y < 0.005 and (self.temp_calibration_color):
                color = np.zeros([self.white_dimensions, self.white_dimensions, 3], dtype=np.uint8)
                color[:] = [self.temp_calibration_color[self.prev_user_choice]]
                time_since_last_blink = time.time()-self.last_selecting
                if time_since_last_blink < 0.75 and time_since_last_blink> 0.25:
                    print("Selecting: ", self.temp_calibration_color[self.prev_user_choice])
                    self.blink_count += 1
                else:
                    self.blink_count = 0
                if self.blink_count == 2:
                    self.current_keyboard_index = (self.current_keyboard_index+1)%4 # 4 is probably not the right number...
                self.last_selecting = time.time()
                cv2.imshow("window_name", color)
            return landmark_points_list
        return None
    def closest_calibration_point(self, pupil):
        """Using the pupil calculated from landmark_to_pupil_avg returns the closest calibration point and the distance to that point.
        Used to actually track what the user is looking at."""
        minimum_distance = 10000
        minimum_index = None
        if (pupil is None) or (not self.gaze_calibration):
            return None
        if len(self.gaze_calibration)-1 == 0:
            return [0, 0]
        for i in range(0, len(self.gaze_calibration)):
            x, y = self.gaze_calibration[i]
            calib_x, calib_y = pupil
            calibration_distance =math.sqrt((x-calib_x)**2+(y-calib_y)**2)
            if calibration_distance < minimum_distance:
                minimum_index = i
                minimum_distance = calibration_distance
        return [minimum_index, minimum_distance]
    def color_calibration_window(self, closest_calibration, window_name="Color_window"):
        """Easiest way to test any number of calibrated points. Uses the index of the closest_calibration_point to choose the
        associated color to the calibration point.  Two seperate lists but ultimately the association never changes."""
        if (closest_calibration is None) or (not closest_calibration):
            return
        self.temp_calibration_color_vote[closest_calibration[0]] += 1
        if (time.time()-self.time_tracking) < .5:
            return
        self.time_tracking = time.time()
        max = 0
        index=None
        for i in range(0, len(self.temp_calibration_color_vote)):
            if self.temp_calibration_color_vote[i] > max:
                max = self.temp_calibration_color_vote[i]
                index = i
            self.temp_calibration_color_vote[i] = 0
        color = np.zeros([self.white_dimensions, self.white_dimensions, 3], dtype=np.uint8)
        self.prev_user_choice = index
        color[:] = [self.temp_calibration_color[index]]
        cv2.imshow(window_name, color)
        """if (closest_calibration is None) or (not closest_calibration):
            return
        color = np.zeros([self.white_dimensions, self.white_dimensions, 3], dtype=np.uint8)
        color[:] = [self.temp_calibration_color[closest_calibration[0]]]
        cv2.imshow(window_name, color)
        return""" 
    def edit_calibration_list(self, begin_point, end_point, amount_split=2):
        """Dead function, doesn't work.  Will probably delete"""
        if begin_point is None or end_point is None or amount_split == 0:
            return
        begin_x, begin_y = begin_point
        end_x, end_y = end_point
        
        for i in range(0, amount_split):
            calc_x = (begin_x-end_x)/amount_split
            calc_y = (begin_y-end_y)/amount_split
            x = end_x+calc_x*(i+1)
            y = end_y+calc_y*(i+1)
            self.gaze_calibration.append([x, y])
            self.temp_calibration_color.append([np.uint8(random.randint(0,255)), 
                                                np.uint8(random.randint(0,255)), 
                                                np.uint8(random.randint(0,255))])
        return
    def draw_letters(self):
        """Helper function to draw the keyboard"""
        char_array=[['A', 'B', 'C', 'D'],['H', 'I', 'J', 'K']]

        for i in range(0, len(char_array)):
            for j in range(0,len(char_array[i])):
                self.white=cv2.putText(self.white, char_array[i][j], (100+j*225, 100+i*400), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
        return
    def draw_eye_circle(self, pupil):
        if pupil == None:
            return
        x1, y1 = pupil
        if self.avg_eye_position_time is None:
            self.avg_eye_position_time = time.time()
        if time.time() - self.avg_eye_position_time > 0.5:
            self.avg_eye_position_time = time.time()
            self.avg_eye_position.clear()
            return
        
        self.avg_eye_position.append(pupil)
        sum_y = 0
        for num in self.avg_eye_position:
            _, num_y = num
            sum_y+=num_y
        y1 = sum_y/len(self.avg_eye_position)
        yz = (y1-self.eye_drawing_calibration[0][1])*(700-100)/(self.eye_drawing_calibration[1][1]-self.eye_drawing_calibration[0][1])
        # xz = (x1-self.eye_drawing_calibration[2]][1])*(700-100)/(self.eye_drawing_calibration[3][1]-self.eye_drawing_calibration[2][1])
        self.white=cv2.circle(self.white, (int(self.white_dimensions/2), int(yz)), 10, (0, 0, 0))
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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x,y,w,h) in faces:
                the_face = self.frame[y:int(y+h),x:x+w]
                landmark_points_list = self.landmark_calculations(self.frame, x, y)
                discovered_pupil = self.landmark_to_pupil_avg(landmark_points_list)
                closest_calibration = self.closest_calibration_point(discovered_pupil)
                self.color_calibration_window(closest_calibration)
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if len(self.eye_drawing_calibration) >= 2:
               self.draw_eye_circle(discovered_pupil)
            self.draw_letters()
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
            elif pressedKey == ord('p'):
                if len(self.gaze_calibration) != 2:
                    continue
                self.edit_calibration_list(self.gaze_calibration[0], self.gaze_calibration[1], amount_split=2)
            elif pressedKey == ord('f'):
                self.color_calibration_window(closest_calibration, window_name="Choice result")
            elif pressedKey == ord('g'):
                self.eye_drawing_calibration.append(discovered_pupil)
            if self.calibrating == True:
                self.calc_calibration_point(discovered_pupil)

if __name__ == "__main__":
    eyes = eye_detection()
    eyes.main_loop()

