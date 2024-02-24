import streamlit as st
from streamlit_webrtc import VideoTransformer, webrtc_streamer
import mediapipe as mp
import cv2
import numpy as np
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define a custom VideoTransformer class for hand detection and finger tracking
class HandDetectionTransformer(VideoTransformer):
    def __init__(self):
        self.tap_count = 0
        self.hand_start_position = None
        self.tap_detected = False
        self.initial_touch_threshold = 20
        self.separation_threshold = 20
        self.tap_cooldown = 0.2
        self.tap_data = []
        self.start_time = time.time()
        self.speeds_graph = []

    def transform(self, frame):
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
                    self.detect_finger_taps(frame, index_tip, thumb_tip)

                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

        return frame

    def detect_finger_taps(self, frame, index_tip, thumb_tip):
        distance_separation = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))

        if not self.tap_detected and distance_separation < self.initial_touch_threshold:
            self.tap_detected = True
            self.hand_start_position = index_tip

        if self.tap_detected and distance_separation > self.separation_threshold:
            self.tap_detected = False
            self.tap_count += 1
            print(f"Tap {self.tap_count} detected! Distance: {distance_separation}")

            # Save data
            self.tap_data.append({
                'Tap Count': self.tap_count,
                'Time': time.time() - self.start_time,
                'Distance (pixels)': distance_separation,
                'Start Position': self.hand_start_position
            })

            # Update the real-time graph
            self.speeds_graph.append(distance_separation)

        # Display real-time graph
        st.line_chart(self.speeds_graph)

# Main Streamlit app
def main():
    st.title("Real-time Hand Detection and Finger Tracking")

    webrtc_ctx = webrtc_streamer(
        key="hand-detection",
        video_transformer_factory=HandDetectionTransformer,
        async_transform=True,
    )

if __name__ == "__main__":
    main()
