# This is a computer vision program developed as part of Inspection Drone Reseach - Navigation Team
# Developed by Adhitya Rizky Andhira Kusumah
# Drone Camera Version

# Step 1: Import modules
import cv2, os, time
import mediapipe as mp
import tensorflow as tf
import numpy as np
from tensorflow import keras
from djitellopy import tello

# Step 2: Initiate algorithm
HandDetect = mp.solutions.hands
Hands = HandDetect.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

PoseDetect = mp.solutions.pose
Pose = PoseDetect.Pose(min_detection_confidence = 0.5)

class RecognizeGesture:
    def __init__(self, model, Drone):
        self.IsCommand = False
        self.Follow = False
        Control = DroneControl()

        #CamFeed = cv2.VideoCapture(0)

        i = 1
        while True:
            if i%24:

                # Capture camera feed
                #ret, frame0 = CamFeed.read()
                frame0 = Drone.get_frame_read().frame
                frame0 = cv2.resize(frame0, (720, 480))

                frame = frame0.copy()
                #frame = cv2.flip(frame, 1)

                # Convert to RGB Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = Hands.process(frame)

                # Check hand
                blk = self.CheckHands(results, frame, mpDraw, HandDetect, i)

                # Drone command
                if self.IsCommand:
                    Control.Go(frame, blk, model, frame, Drone)
                    if Control.Follow:
                        if not self.Follow:
                            self.Follow = True
                        elif self.Follow:
                            self.Follow = False
                if self.Follow:
                    Control.FollowMe(frame)
                    self.Follow = Control.Follow

                Drone.send_rc_control(0, 0, 0, 0)

                # Display feed
                cv2.putText(frame, 'Battery: ' + str(Drone.get_battery()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('Feed', frame)

                # Quit button = q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    Drone.streamoff()
                    break

            self.IsCommand = False

            i += 1

        # Release capture object
        #CamFeed.release()

        # Destroy all windows
        cv2.destroyAllWindows()

    def CheckHands(self, results, frame, mpDraw, HandDetect, frame_num):
        h, w, c = frame.shape
        b, t, l, r = 0, h, 0, w

        blk = np.zeros((h, w, c))

        if results.multi_hand_landmarks:
            for handsLms in results.multi_hand_landmarks:
                x_max, y_max, x_min, y_min = 0, 0, w, h
                for lm in handsLms.landmark:
                    x, y = int(lm.x*w), int(lm.y*h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                mpDraw.draw_landmarks(blk, handsLms, HandDetect.HAND_CONNECTIONS)

            # Define boundary box size and centerpoint
            rad = max([y_max-y_min, x_max-x_min])
            x_ctp, y_ctp = int((x_max+x_min)/2), int((y_max+y_min)/2)

            if (y_ctp-rad) < 0 or (x_ctp-rad) < 0:
                rad = min([y_ctp, x_ctp])
            if (y_ctp+rad) > h or (x_ctp+rad) > w:
                rad = min([h-y_ctp, w-x_ctp])
            
            b, t, l, r = y_ctp-rad, y_ctp+rad, x_ctp-rad, x_ctp+rad

            # Image cropping
            blk = blk[b:t, l:r]
            blk = cv2.resize(blk, (256, 256))

            print(frame_num)

            self.IsCommand = True

        return blk

class DroneControl:
    def __init__(self):
        self.Follow = False
        self.FirstFollow = True
        self.p1 = self.p2 = self.p3 = self.p4 = self.Area0 = 0

    def Go(self, out_frame, hands, model, frame, Drone):
        detection_prob = model.predict(tf.expand_dims(hands, axis=0))
        MaxVal = np.max(detection_prob, axis=-1)
        if MaxVal >= 0.95:
            prediction = int(np.argmax(detection_prob, axis=-1))
            self.Result = self.PredictionResults(prediction)

            h = out_frame.shape[0]
            cv2.putText(out_frame, self.Result, (10, round(h)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            self.ControlCommand(self.Result, frame, out_frame, Drone)

    def PredictionResults(self, result):
        Pred = {
            0   : 'Land',
            1   : 'Move Backward',
            2   : 'Move Down',
            3   : 'Move Forward',
            4   : 'Move Left',
            5   : 'Move Right',
            6   : 'Move Up',
            7   : 'Takeoff',
            8   : 'Yaw Left',
            9   : 'Yaw Right'
        }

        return Pred[result]

    def ControlCommand(self, result, frame, out_frame, Drone):
        self.Throttle = self.Pitch = self.Roll = self.Yaw = 0

        if result == 'Takeoff' and (not Drone.is_flying):
            Drone.takeoff()
        elif result == 'Takeoff' and (Drone.is_flying):
            if not self.Follow:
                self.Follow = True
                cv2.putText(out_frame, "Start Following", (10, round(out_frame.shape[0])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            elif self.Follow:
                self.Follow = False
                self.FirstFollow = True
                cv2.putText(out_frame, "Stop Following", (10, round(out_frame.shape[0])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            time.sleep(2)

        elif result == 'Land' and (Drone.is_flying):
            Drone.land()
        elif result == 'Move Up' and (Drone.is_flying):
            self.Throttle = 40
        elif result == 'Move Down' and (Drone.is_flying):
            self.Throttle = -40
        elif result == 'Move Forward' and (Drone.is_flying):
            self.Pitch = 40
        elif result == 'Move Backward' and (Drone.is_flying):
            self.Pitch = -40
        elif result == 'Move Right' and (Drone.is_flying):
            self.Roll = 40
        elif result == 'Move Left' and (Drone.is_flying):
            self.Roll = -40
        elif result == 'Yaw Right' and (Drone.is_flying):
            self.Yaw = 40
        elif result == 'Yaw Left' and (Drone.is_flying):
            self.Yaw = -40
        else:
            self.Throttle = self.Pitch = self.Roll = self.Yaw = 0

        Drone.send_rc_control(self.Roll, self.Pitch, self.Throttle, self.Yaw)

    def FollowMe(self, frame):
        h, w = frame.shape[0:2]

        self.Throttle = self.Pitch = self.Roll = self.Yaw = 0
        
        PoseRes = Pose.process(frame)
        if PoseRes.pose_landmarks:
            if (PoseRes.pose_landmarks.landmark[PoseDetect.PoseLandmark.LEFT_SHOULDER].visibility > 0.95):
                x2, y2, x1 = PoseRes.pose_landmarks.landmark[PoseDetect.PoseLandmark.LEFT_SHOULDER].x, PoseRes.pose_landmarks.landmark[PoseDetect.PoseLandmark.LEFT_SHOULDER].y, PoseRes.pose_landmarks.landmark[PoseDetect.PoseLandmark.RIGHT_SHOULDER].x
                if x1 > x2:
                    x1, x2 = x2, x1
                    y2 = PoseRes.pose_landmarks.landmark[PoseDetect.PoseLandmark.RIGHT_SHOULDER].y
                y1 = y2 - (x2 - x1)

                if self.FirstFollow:
                    self.p1, self.p2, self.p3, self.p4 = 0.2, 0.5, 0.5, 0.8
                    cv2.rectangle(frame, (int((self.p1)*w), int((self.p2)*h)), (int((self.p3)*w), int((self.p4)*h)), (255, 0, 0), 3)
                    self.Area0 = (self.p3-self.p1)*(self.p4-self.p2)*w*h
                    self.FirstFollow = False
                
                cv2.rectangle(frame, (int((x1)*w), int((y1)*h)), (int((x2)*w), int((y2)*h)), (0, 255, 0), 3)
                cv2.rectangle(frame, (int((self.p1)*w), int((self.p2)*h)), (int((self.p3)*w), int((self.p4)*h)), (255, 0, 0), 3)
                Area = (x2-x1)*(y2-y1)*w*h
        
                if (abs((x1 - self.p1)*w) <= 40):
                    self.Roll = 0
                elif (x1 < self.p1):
                    self.Roll = -20
                    cv2.putText(frame, 'Move Left', (10, round(h)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                elif (x1 > self.p1):
                    self.Roll = 20
                    cv2.putText(frame, 'Move Right', (10, round(h)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
                if (abs((y1 - self.p2)*h) <= 40):
                    self.Throttle = 0
                elif (y1 < self.p2):
                    self.Throttle = 20
                    cv2.putText(frame, 'Move Up', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                elif (y1 > self.p2):
                    self.Throttle = -20
                    cv2.putText(frame, 'Move Down', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if (abs((Area/self.Area0) - 1) <= 0.1):
                    self.Pitch = 0
                elif (Area < self.Area0):
                    self.Pitch = 20
                    cv2.putText(frame, 'Move Forward', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                elif (Area > self.Area0):
                    self.Pitch = -20
                    cv2.putText(frame, 'Move Backward', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        Drone.send_rc_control(self.Roll, self.Pitch, self.Throttle, self.Yaw)

class Activate:
    def __init__(self, Drone):
        self.Running = True
        Drone.streamon()

        print('Loading model...')
        try:
            self.model_path = os.path.join(os.path.dirname(__file__), 'MODEL J')
            self.model = keras.models.load_model(self.model_path)
            self.model.trainable = False
            print('Load success!')
        except:
            print('ERROR! Model not found!')
            self.Running = False

    def start(self, Drone):
        if not self.Running:
            self.EndProgram(Drone)
        if self.Running:
            RecognizeGesture(self.model, Drone)
            self.EndProgram(Drone)

    def EndProgram(self, Drone):
        self.Running = False

        print('Closing...')
        Drone.streamoff()
        Drone.end()

        print('Program ends here!')

# Step XX: Run program
print('Program starts here.\n')

try:
    Drone = tello.Tello()
    Drone.connect()
    print('Drone connected!')

    HandNav = Activate(Drone)
    HandNav.start(Drone)
except:
    pass