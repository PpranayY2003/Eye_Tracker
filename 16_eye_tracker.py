# mp.solutions:
# This is the main namespace of Mediapipe.
# It contains prebuilt solutions for tasks like face detection, hand tracking, pose estimation, etc.

# mp.solutions.face_mesh:
# The face_mesh module in Mediapipe provides functionality for detecting and tracking facial landmarks (468 specific points on the face). This is commonly used in augmented reality, filters, and facial analysis.

# mp.solutions.face_mesh.FaceMesh:
# The FaceMesh class is used to initialize the face mesh model. It provides methods to process images and return facial landmarks.

# mp.solutions.face_mesh.FaceMesh:
# The FaceMesh class is used to initialize the face mesh model. It provides methods to process images and return facial landmarks.
import numpy as np
import cv2 as cv
import mediapipe as mp
import pyautogui
import time
import os
import tensorflow as tf

# Suppress TensorFlow Lite logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

# Initialize Mediapipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
cam = cv.VideoCapture(0)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cam.read()
    frame_count += 1
    frame = cv.flip(frame, 1)

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Measure processing time per frame
    process_start_time = time.time()
    output = face_mesh.process(rgb_frame)
    process_end_time = time.time()
    response_time = (process_end_time - process_start_time) * 1000

    landmark_points = output.multi_face_landmarks  
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            if id == 1:
                screen_x = int(screen_w * landmark.x)
                screen_y = int(screen_h * landmark.y)
                pyautogui.moveTo(screen_x, screen_y)
        
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv.circle(frame, (x, y), 2, (200, 0, 0), -1)
            if (left[0].y - left[1].y) < 0.005:
                pyautogui.click()
                pyautogui.sleep(1)

    if time.time() - start_time > 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()
        print(f"FPS: {fps}, Response Time: {response_time:.2f} ms")
    
    cv.imshow("Face Mesh", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
