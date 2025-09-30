import streamlit as st
import cv2
import os, tempfile, queue, json
import sounddevice as sd
import vosk
from deepface import DeepFace
from PIL import Image
from datetime import datetime

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="AstroMaitri - AI Assistant",
    page_icon="🚀",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #a8dadc;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #f1faee;
            margin-bottom: 30px;
        }
        .user-box {
            background-color: #1d3557;
            padding: 14px;
            border-radius: 12px;
            margin: 10px 0;
            color: #f1faee;
        }
        .astromaitri-box {
            background-color: #457b9d;
            padding: 14px;
            border-radius: 12px;
            margin: 10px 0;
            color: #f1faee;
            font-weight: 500;
        }
        .emotion-box {
            background-color: #2a9d8f;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            margin: 10px 0;
            color: #fff;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Initialize Vosk model
# -----------------------
if "vosk_model" not in st.session_state:
    st.session_state.vosk_model = vosk.Model("vosk-model-small-en-us-0.15")  # keep in project folder

model = st.session_state.vosk_model
samplerate = 16000
q = queue.Queue()

def record_audio(duration=5):
    """Record audio for a few seconds and return transcript."""
    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                           dtype='int16', channels=1, callback=callback):
        st.info("🎤 Recording... (Speak now)")
        rec = vosk.KaldiRecognizer(model, samplerate)
        transcript_text = ""

        for _ in range(int(duration * samplerate / 8000)):
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcript_text += " " + result.get("text", "")

        final_result = json.loads(rec.FinalResult())
        transcript_text += " " + final_result.get("text", "")
        return transcript_text.strip()

# -----------------------
# Streamlit UI
# -----------------------
st.markdown("<h1 class='title'>🌌 AstroMaitri</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your AI Companion for Psychological & Physical Well-Being</p>", unsafe_allow_html=True)

if st.button("✨ Start Interaction"):
    col1, col2 = st.columns(2)  # Left & Right layout

    with col1:
        # ---- 1. Capture face ----
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            cv2.imwrite(temp_path, frame)

            st.image(frame, channels="BGR", caption="📸 Captured Image")

            try:
                result = DeepFace.analyze(temp_path, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                st.markdown(f"<div class='emotion-box'>🧠 Detected Emotion: {emotion.capitalize()}</div>", unsafe_allow_html=True)
            except Exception as e:
                emotion = "unknown"
                st.error(f"Emotion detection failed: {e}")
        else:
            emotion = "unknown"
            st.error("Could not access webcam.")

    with col2:
        # ---- 2. Record speech ----
        transcript = record_audio(duration=5)
        if transcript:
            st.markdown(f"<div class='user-box'>👨‍🚀 You said: {transcript}</div>", unsafe_allow_html=True)
        else:
            transcript = ""
            st.warning("⚠️ No speech detected.")

    # ---- 3. AstroMaitri response ----
    responses = {
        "happy": "Great to see you smiling! Keep it up 🌞",
        "sad": "I’m here with you. Take a deep breath 💙",
        "angry": "Try to relax. Let's slow down a bit 🙏",
        "neutral": "All steady. You're doing well 🚀",
    }
    reply = responses.get(emotion, "Stay strong, you're doing great.")

    negative_words = ["tired", "angry", "sad", "depressed", "lonely", "stressed"]
    if any(word in transcript.lower() for word in negative_words):
        reply = "⚠️ Alert: Crew member shows signs of stress. Reporting to Ground Control."

    st.markdown(f"<div class='astromaitri-box'>🤖 AstroMaitri: {reply}</div>", unsafe_allow_html=True)
