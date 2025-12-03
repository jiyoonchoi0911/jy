import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
import av
import time
import base64
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

# --- CSS & Audio Script ---
def get_audio_html(sound_file_path):
    # Use browser built-in audio for reliability
    js_code = """
        <script>
        function playAlert() {
            var audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
            audio.volume = 0.5;
            audio.play();
        }
        </script>
        <div id="audio-container"></div>
    """
    return js_code

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold; font-size: 30px; }
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 30px; }
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 30px; animation: blink 1s infinite; }
    
    .advice-box {
        background-color: #fff9c4;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #fbc02d;
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-top: 10px;
    }

    @keyframes blink {
        50% { opacity: 0.5; }
    }
    
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(get_audio_html(""), unsafe_allow_html=True)

st.title("🐢 AI Posture Correction Pro")
st.markdown("Turn on the webcam to analyze your posture. **Follow the guide below to improve your posture.**")

# --- Load Model & MediaPipe ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()
mp_pose = mp.solutions.pose

# --- Session State (Calibration) ---
if 'calibration_y' not in st.session_state:
    st.session_state.calibration_y = 0.0

# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model
        
        # 1. Smoothing
        self.history_len = 10
        self.prob_history = deque(maxlen=self.history_len)
        
        # Shared variables
        self.latest_probs = {'good': 0, 'mild': 0, 'severe': 0}
        self.latest_pred = "good"
        self.severe_consecutive_frames = 0
        self.trigger_sound = False
        self.cal_y = 0 

    def update_calibration(self, value):
        self.cal_y = value

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        current_pred = "good"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # 2. Feature Extraction
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                
                # Apply Calibration (Simple Y-offset simulation if needed in future)
                # center_y += self.cal_y 

                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1

                indices = [0, 2, 5, 7, 8, 11, 12]
                features = []
                keypoints = {}

                for idx in indices:
                    lm = landmarks[idx]
                    norm_x = (lm.x - center_x) / width
                    norm_y = (lm.y - center_y) / width
                    features.extend([norm_x, norm_y])
                    px, py = int(lm.x * w), int(lm.y * h)
                    keypoints[idx] = (px, py)

                # 3. Prediction
                if self.model:
                    probs = self.model.predict_proba([features])[0]
                    self.prob_history.append(probs)
                    
                    avg_probs = np.mean(self.prob_history, axis=0)
                    classes = self.model.classes_
                    prob_dict = {cls: p for cls, p in zip(classes, avg_probs)}
                    self.latest_probs = prob_dict
                    
                    current_pred = max(prob_dict, key=prob_dict.get)
                    self.latest_pred = current_pred
                    
                    # 4. Visualization (Skeleton)
                    color = (0, 255, 0) # Green
                    if current_pred == 'mild': color = (0, 255, 255) # Yellow
                    if current_pred == 'severe': color = (0, 0, 255) # Red

                    # Draw Points
                    for idx, (px, py) in keypoints.items():
                        cv2.circle(img, (px, py), 5, color, -1)
                    
                    # Draw Skeleton (Shoulders & Neck)
                    if 11 in keypoints and 12 in keypoints:
                        cv2.line(img, keypoints[11], keypoints[12], color, 2)
                    if 0 in keypoints:
                        sh_center = ((keypoints[11][0] + keypoints[12][0]) // 2, 
                                     (keypoints[11][1] + keypoints[12][1]) // 2)
                        cv2.line(img, sh_center, keypoints[0], color, 2)

                    # 5. Sound Logic (Trigger if severe for ~1 sec)
                    if current_pred == 'severe':
                        self.severe_consecutive_frames += 1
                        if self.severe_consecutive_frames > 30:
                            self.trigger_sound = True
                    else:
                        self.severe_consecutive_frames = 0
                        self.trigger_sound = False
                        
            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
col_main, col_sidebar = st.columns([3, 1])

with col_main:
    # Calibration Button
    if st.button("📏 Set Current Posture as 'Standard' (Reset)"):
        st.session_state.calibration_y = 0.0 # Placeholder for future logic
        st.success("Calibration Set! (Logic ready)")

    if model is None:
        st.error("Model file (posture_model.pkl) is missing.")
    else:
        ctx = webrtc_streamer(
            key="posture-pro",
            video_processor_factory=VideoProcessor,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

with col_sidebar:
    st.markdown("### 📊 Live Status")
    status_ph = st.empty()
    advice_ph = st.empty()
    
    st.write("---")
    st.markdown("### Posture Score")
    # Only one progress bar for simplicity
    score_ph = st.empty()
    
    # Hidden placeholder for sound
    sound_ph = st.empty()

# --- Main Loop ---
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            probs = ctx.video_processor.latest_probs
            pred = ctx.video_processor.latest_pred
            trigger_sound = ctx.video_processor.trigger_sound
            
            # 1. Update Status Text & Advice
            if pred == 'good':
                status_ph.markdown(f"<div class='good-text'>GOOD 😊</div>", unsafe_allow_html=True)
                advice_ph.markdown(f"<div class='advice-box'>✅ Perfect alignment! Keep it up.</div>", unsafe_allow_html=True)
            
            elif pred == 'mild':
                status_ph.markdown(f"<div class='mild-text'>MILD 😐</div>", unsafe_allow_html=True)
                advice_ph.markdown(f"<div class='advice-box'>💡 Lift your head slightly.<br>Relax your shoulders.</div>", unsafe_allow_html=True)
            
            else: # severe
                status_ph.markdown(f"<div class='severe-text'>SEVERE 🐢</div>", unsafe_allow_html=True)
                advice_ph.markdown(f"<div class='advice-box'>🚨 <b>Pull chin back!</b><br>Open your chest.</div>", unsafe_allow_html=True)
            
            # 2. Update Single Posture Score Bar (Probability of Good)
            good_score = int(probs.get('good', 0) * 100)
            score_ph.progress(good_score, text=f"{good_score}%")

            # 3. Sound Alert
            if trigger_sound:
                sound_ph.markdown("""
                    <script>
                    playAlert();
                    </script>
                    """, unsafe_allow_html=True)
            else:
                sound_ph.empty()

        time.sleep(0.1)
