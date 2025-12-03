import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import time
import base64
import os
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

# --- Audio Handling (File Based) ---
def get_audio_html(file_path):
    """
    Reads a local audio file and returns an HTML string to play it automatically.
    This avoids using hardcoded base64 strings in the code.
    """
    if not os.path.exists(file_path):
        return ""
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    # Unique ID based on time to force browser to re-render audio tag
    unique_id = time.time() 
    
    return f"""
        <audio autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        <div style="display:none;">{unique_id}</div>
    """

# --- CSS & Voice Script ---
def get_voice_script():
    js_code = """
        <script>
        // ---- Voice guidance using Web Speech API ----
        window.lastPostureStatus = null;
        function speakPostureStatus(status) {
            if (!('speechSynthesis' in window)) return;
            var text = '';
            if (status === 'GOOD') { text = 'Posture is good.'; } 
            else if (status === 'MILD') { text = 'Posture is mild.'; } 
            else if (status === 'SEVERE') { text = 'Posture is severe.'; }
            if (text === '') return;
            var utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }
        function updatePostureStatus(status) {
            if (window.lastPostureStatus === status) return;
            window.lastPostureStatus = status;
            speakPostureStatus(status);
        }
        </script>
    """
    return js_code

st.markdown("""
    <style>
    .good-text { color: #2ecc71; font-weight: bold; font-size: 40px; text-align: center; }
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 40px; text-align: center; }
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 40px; text-align: center; animation: blink 1s infinite; }
    .advice-box { background-color: #fff9c4; padding: 15px; border-radius: 10px; border-left: 5px solid #fbc02d; font-size: 18px; font-weight: bold; color: #333; margin-top: 10px; margin-bottom: 20px; }
    @keyframes blink { 50% { opacity: 0.5; } }
    .stProgress > div > div > div > div { background-color: #2ecc71; }
    </style>
    """, unsafe_allow_html=True)

st.markdown(get_voice_script(), unsafe_allow_html=True)

st.title("🐢 AI Posture Correction Pro")
st.markdown("**Step 1:** Sit straight. **Step 2:** Click 'Set Standard'. **Step 3:** Keep that posture.")

mp_pose = mp.solutions.pose

# --- Distance → Probabilities ---
def distance_to_probs(distance, t_good=0.12, t_mild=0.23):
    d = float(distance)
    good_score = max(0.0, 1.0 - d / max(t_good, 1e-6))
    if d <= t_good: mild_score = d / max(t_good, 1e-6)
    elif d <= t_mild: mild_score = 1.0 - (d - t_good) / max(t_mild - t_good, 1e-6)
    else: mild_score = 0.0
    if d <= t_mild: severe_score = 0.0
    else: severe_score = min(1.0, (d - t_mild) / max(t_mild, 1e-6))
    scores = {"good": good_score, "mild": mild_score, "severe": severe_score}
    total = sum(scores.values())
    if total <= 0: return {"good": 1/3, "mild": 1/3, "severe": 1/3}
    for k in scores: scores[k] /= total
    return scores

# --- Feature extraction ---
def extract_features_from_landmarks(landmarks, img_shape):
    l_sh = landmarks[11]; r_sh = landmarks[12]
    center_x = (l_sh.x + r_sh.x) / 2.0; center_y = (l_sh.y + r_sh.y) / 2.0
    width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
    if width == 0: width = 1.0
    indices = [0, 2, 5, 7, 8, 11, 12]
    features = []; h, w, _ = img_shape; keypoints = {}
    for idx in indices:
        lm = landmarks[idx]
        norm_x = (lm.x - center_x) / width; norm_y = (lm.y - center_y) / width
        features.extend([norm_x, norm_y])
        px, py = int(lm.x * w), int(lm.y * h); keypoints[idx] = (px, py)
    return features, keypoints

# --- Video Processor ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.baseline = None; self.calibrate_now = False
        self.distance_history = deque(maxlen=10)
        self.latest_probs = {"good": 0.0, "mild": 0.0, "severe": 0.0}
        self.latest_pred = "good"; self.latest_distance = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24"); h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                features, keypoints = extract_features_from_landmarks(landmarks, img.shape)
                if self.calibrate_now:
                    self.baseline = np.array(features); self.distance_history.clear(); self.calibrate_now = False
                if self.baseline is not None:
                    diff = np.array(features) - np.array(self.baseline)
                    dist = float(np.linalg.norm(diff))
                    self.distance_history.append(dist)
                    avg_dist = float(np.mean(self.distance_history))
                    self.latest_distance = avg_dist
                    prob_dict = distance_to_probs(avg_dist)
                    self.latest_probs = prob_dict
                    self.latest_pred = max(prob_dict, key=prob_dict.get)
                else:
                    self.latest_distance = 0.0
                    self.latest_probs = {"good": 1.0, "mild": 0.0, "severe": 0.0}
                    self.latest_pred = "good"
                current_pred = self.latest_pred
                color = (0, 255, 0)
                if current_pred == "mild": color = (0, 255, 255)
                elif current_pred == "severe": color = (0, 0, 255)
                for _, (px, py) in keypoints.items(): cv2.circle(img, (px, py), 5, color, -1)
                if 11 in keypoints and 12 in keypoints: cv2.line(img, keypoints[11], keypoints[12], color, 2)
                if 0 in keypoints and 11 in keypoints and 12 in keypoints:
                    sh_center = ((keypoints[11][0] + keypoints[12][0]) // 2, (keypoints[11][1] + keypoints[12][1]) // 2)
                    cv2.line(img, sh_center, keypoints[0], color, 2)
            except Exception: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Layout ---
col_video, col_info = st.columns([3, 2])
ctx = None

with col_video:
    st.subheader("Webcam")
    ctx = webrtc_streamer(
        key="posture-pro",
        video_processor_factory=VideoProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown("---")
    calib_msg_ph = st.empty()
    if st.button("📏 Set Current Posture as Standard", use_container_width=True):
        if ctx and ctx.video_processor:
            ctx.video_processor.calibrate_now = True
            calib_msg_ph.success("✅ Standard posture set!")
        else: calib_msg_ph.warning("Wait for webcam.")

with col_info:
    st.markdown("### 📊 Live Status")
    status_ph = st.empty(); advice_ph = st.empty()
    st.markdown("---")
    st.markdown("### Posture Scores")
    st.write("Good:"); bar_good_ph = st.empty()
    st.write("Mild:"); bar_mild_ph = st.empty()
    st.write("Severe:"); bar_severe_ph = st.empty()
    st.markdown("---"); dist_ph = st.empty()
    sound_ph = st.empty(); tts_ph = st.empty()

# --- Main Loop & Sound Logic ---
last_sound_time = 0
SOUND_INTERVAL = 2.0 # Seconds
SOUND_FILE = "alert.mp3" # Your local sound file name

if ctx and ctx.state.playing:
    while True:
        if not ctx.state.playing: break
        vp = ctx.video_processor
        if vp is not None:
            probs = vp.latest_probs; pred = vp.latest_pred; dist = vp.latest_distance
            
            # Status
            if pred == "good":
                status_ph.markdown("<div class='good-text'>GOOD 😊</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>✅ Perfect! Keep it up.</div>", unsafe_allow_html=True)
                tts_ph.markdown("<script>updatePostureStatus('GOOD');</script>", unsafe_allow_html=True)
            elif pred == "mild":
                status_ph.markdown("<div class='mild-text'>MILD 😐</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>💡 Lift head slightly.<br>Relax shoulders.</div>", unsafe_allow_html=True)
                tts_ph.markdown("<script>updatePostureStatus('MILD');</script>", unsafe_allow_html=True)
            else:
                status_ph.markdown("<div class='severe-text'>SEVERE 🐢</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>🚨 <b>Pull chin back!</b><br>Open chest.</div>", unsafe_allow_html=True)
                tts_ph.markdown("<script>updatePostureStatus('SEVERE');</script>", unsafe_allow_html=True)

            # Scores
            g, m, s = probs.get("good", 0.0)*100, probs.get("mild", 0.0)*100, probs.get("severe", 0.0)*100
            bar_good_ph.progress(int(g), text=f"{g:.1f}%")
            bar_mild_ph.progress(int(m), text=f"{m:.1f}%")
            bar_severe_ph.progress(int(s), text=f"{s:.1f}%")
            dist_ph.markdown(f"Deviation: **{dist:.3f}**")

            # --- Sound Logic (Reads 'alert.mp3') ---
            if pred == "severe":
                current_time = time.time()
                if current_time - last_sound_time > SOUND_INTERVAL:
                    # Check if file exists to prevent errors
                    if os.path.exists(SOUND_FILE):
                        sound_html = get_audio_html(SOUND_FILE)
                        sound_ph.markdown(sound_html, unsafe_allow_html=True)
                        last_sound_time = current_time
                    else:
                        # Optional: Alert user if file is missing (only once to avoid spam)
                        # sound_ph.warning(f"File '{SOUND_FILE}' not found.")
                        pass
            else:
                sound_ph.empty()

        time.sleep(0.1)

