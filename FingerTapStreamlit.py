import cv2
import mediapipe as mp
import numpy as np
import csv
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import time
import streamlit as st

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# List to store tap data
tap_data = []


class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Process hand landmarks and detect taps
        frm = detect_hands(frm)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')


def detect_hands(frame):
    global tap_data

    # Convert BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) >= 21:
                index_tip = lm_list[8][1], lm_list[8][2]
                thumb_tip = lm_list[4][1], lm_list[4][2]

                cv2.circle(frame, index_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, thumb_tip, 10, (0, 255, 0), cv2.FILLED)

                # Detect finger taps
                detect_finger_taps(index_tip, thumb_tip)

                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    return frame


def detect_finger_taps(index_tip, thumb_tip):
    global tap_data

    # Calculate distance between index and thumb tips
    distance_separation = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))

    # Define tap detection thresholds
    initial_touch_threshold = 20

    # Detect tap
    if distance_separation < initial_touch_threshold:
        tap_data.append({
            'Time': time.time(),
            'Distance (pixels)': distance_separation,
            'Start Position': index_tip
        })


rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Streamlit app
st.title("Real-time Hand Detection and Finger Tracking")

webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
)

if webrtc_ctx.video_processor:
    while True:
        time.sleep(0.1)  # Update the graph every 0.1 seconds

        # Save tap data to CSV file
        with open('tap_data.csv', mode='w', newline='') as file:
            fieldnames = ['Time', 'Distance (pixels)', 'Start Position']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(tap_data)
