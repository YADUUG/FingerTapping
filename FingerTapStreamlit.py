import cv2
import mediapipe as mp
import streamlit as st
import time
import matplotlib.pyplot as plt
from math import hypot
import numpy as np

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Initialize variables
start_time = time.time()
speeds_graph = []
tap_count = 0
tap_data = []
tap_timestamps = []

# Thresholds
initial_touch_threshold = 20  # Adjust sensitivity for initial touch
separation_threshold = 20  # Adjust sensitivity for separation
tap_cooldown = 0.2  # Decreased cooldown for faster tap detection

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return hypot(point1[0] - point2[0], point1[1] - point2[1])

# Function to detect finger taps and update tap count and tap durations
def detect_finger_taps(frame, index_tip, thumb_tip):
    global tap_count, tap_detected, hand_start_position, tap_timestamps

    distance_separation = calculate_distance(index_tip, thumb_tip)

    if not tap_detected and distance_separation < initial_touch_threshold:
        tap_detected = True
        hand_start_position = index_tip
        tap_timestamps.append(time.time())

    if tap_detected and distance_separation > separation_threshold:
        tap_detected = False
        tap_count += 1
        tap_data.append({
            'Tap Count': tap_count,
            'Time': time.time() - start_time,
            'Distance (pixels)': distance_separation,
            'Start Position': hand_start_position
        })

# Function to display real-time graph using Matplotlib
def display_realtime_graph():
    plt.clf()
    plt.plot(speeds_graph)
    plt.title('Finger Tap Distance Over Time')
    plt.xlabel('Frames')
    plt.ylabel('Distance (pixels)')
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Real-time Finger Tap Detection")

    # Get video input
    video_file = st.file_uploader("Upload a video file", type=['mp4'])
    if video_file is not None:
        cap = cv2.VideoCapture(video_file)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            results = hands.process(frame_rgb)

            lmList = []
            index_speed = 0
            thumb_tip = None
            index_tip = None

            if results.multi_hand_landmarks:
                for handlandmark in results.multi_hand_landmarks:
                    for id, lm in enumerate(handlandmark.landmark):
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                    if len(lmList) >= 21:
                        index_tip = lmList[8][1], lmList[8][2]
                        thumb_tip = lmList[4][1], lmList[4][2]

                        cv2.circle(frame, index_tip, 10, (0, 255, 0), cv2.FILLED)
                        cv2.circle(frame, thumb_tip, 10, (0, 255, 0), cv2.FILLED)

                        if index_tip is not None and thumb_tip is not None:
                            # Draw a line between index finger and thumb
                            cv2.line(frame, index_tip, thumb_tip, (255, 0, 0), 2)

                            # Detect finger taps
                            detect_finger_taps(frame, index_tip, thumb_tip)

                mpDraw.draw_landmarks(frame, handlandmark, mpHands.HAND_CONNECTIONS)

            # Display the frame with finger tap detection
            st.image(frame, channels="BGR")

            # Display real-time graph
            display_realtime_graph()

            # Delay between frames
            time.sleep(0.05)

        cap.release()

if __name__ == "__main__":
    main()
