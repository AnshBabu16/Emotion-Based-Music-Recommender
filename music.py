import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import webbrowser

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize Mediapipe holistic model
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Manage session state
if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except FileNotFoundError:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

# Define the Emotion Processor class
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Streamlit UI
st.title("🎵 Emotion-Based Music Recommender 🎶")

# User Inputs
lang = st.text_input("Enter Language (e.g., English, Hindi)")
singer = st.text_input("Enter Singer Name")

# Start Webcam Only If Emotion Hasn't Been Captured
if lang and singer and st.session_state["run"]:
    webrtc_streamer(key="emotion_stream", video_processor_factory=lambda: EmotionProcessor())

# Song Recommendation Button
if st.button("Recommend me songs"):
    if not emotion:
        st.warning("⚠️ Please let me capture your Emotion first!")
        st.session_state["run"] = True
    elif lang and singer:
        st.success(f"✅ Fetching songs for {singer} in {lang}...")
        
        # Placeholder for actual recommendation system
        st.write("🎵 **Song 1** - Example Song by " + singer)
        st.write("🎵 **Song 2** - Another Example by " + singer)

        # Open YouTube with the searched emotion
        search_query = f"{singer} {emotion} {lang} songs"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")

        # Reset Emotion Capture for next use
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False
    else:
        st.warning
