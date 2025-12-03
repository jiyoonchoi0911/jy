import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import time
import base64
import os
import joblib
from PIL import Image
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

# --- Load AI Model (For Photo Upload) ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()

# --- Audio Handling (File Based) ---
def get_audio_html(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
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
st.markdown("**Webcam:** Sit straight & Click 'Set Standard'. **Upload:** Auto-diagnosis using AI.")

mp_pose = mp.solutions.pose

# --- Helper: Distance → Probabilities ---
def distance_to_probs(distance, t_good=0.12, t_mild=0.28):
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

# --- Helper: Feature extraction ---
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

# --- Helper: Draw Visuals ---
def draw_visuals(img, keypoints, pred):
    color = (0, 255, 0)
    if pred == "mild": color = (0, 255, 255)
    elif pred == "severe": color = (0, 0, 255)
    
    for _, (px, py) in keypoints.items(): cv2.circle(img, (px, py), 5, color, -1)
    if 11 in keypoints and 12 in keypoints: cv2.line(img, keypoints[11], keypoints[12], color, 2)
    if 0 in keypoints and 11 in keypoints and 12 in keypoints:
        sh_center = ((keypoints[11][0] + keypoints[12][0]) // 2, (keypoints[11][1] + keypoints[12][1]) // 2)
        cv2.line(img, sh_center, keypoints[0], color, 2)
    return img

# --- Video Processor (Webcam) ---
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
                
                # Draw
                draw_visuals(img, keypoints, self.latest_pred)

            except Exception: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Layout Structure ---
col_left, col_right = st.columns([3, 2])

# Global variables for display
display_probs = {"good": 0.0, "mild": 0.0, "severe": 0.0}
display_pred = "good"
display_dist = 0.0
is_severe_sound = False

# --- Left Column: Input (Webcam or Upload) ---
with col_left:
    tab1, tab2 = st.tabs(["📹 Live Webcam", "🖼️ Upload Photo"])

    # TAB 1: Webcam
    with tab1:
        # Create a container at the TOP for the button
        top_controls = st.container()

        ctx = webrtc_streamer(
            key="posture-pro",
            video_processor_factory=VideoProcessor,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Render Button inside Top Container (Above Video)
        with top_controls:
            calib_msg_ph = st.empty()
            if st.button("📏 Set Current Posture as Standard", use_container_width=True, type="primary"):
                if ctx and ctx.video_processor:
                    ctx.video_processor.calibrate_now = True
                    calib_msg_ph.success("✅ Standard posture set!")
                else: calib_msg_ph.warning("Wait for webcam to start.")
            st.markdown("---")

    # TAB 2: Photo Upload
    with tab2:
        uploaded_file = st.file_uploader("Upload an image for diagnosis", type=['jpg', 'jpeg', 'png'])
        if uploaded_file and model:
            image = Image.open(uploaded_file)
            img_np = np.array(image.convert('RGB'))
            h, w, c = img_np.shape
            
            # MediaPipe Process
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1) as pose:
                results = pose.process(img_np)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    features, keypoints = extract_features_from_landmarks(landmarks, img_np.shape)
                    
                    # AI Model Prediction (No calibration needed for static image)
                    probs = model.predict_proba([features])[0]
                    classes = model.classes_
                    display_probs = {cls: p for cls, p in zip(classes, probs)}
                    display_pred = model.predict([features])[0]
                    
                    # Draw visual on image
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_visual = draw_visuals(img_bgr, keypoints, display_pred)
                    st.image(cv2.cvtColor(img_visual, cv2.COLOR_BGR2RGB), caption="Analysis Result", use_column_width=True)
                else:
                    st.error("⚠️ No person detected in the image.")
        elif uploaded_file and not model:
            st.error("❌ Model file (posture_model.pkl) is missing.")

# --- Logic to Sync Webcam Data to Global Variables ---
if ctx and ctx.state.playing and ctx.video_processor:
    display_probs = ctx.video_processor.latest_probs
    display_pred = ctx.video_processor.latest_pred
    display_dist = ctx.video_processor.latest_distance
    # Trigger sound only on webcam mode logic
    if display_pred == "severe":
        is_severe_sound = True

# --- Right Column: Live Status & Scores (Unified Display) ---
with col_right:
    st.markdown("### 📊 Status Report")
    status_ph = st.empty()
    advice_ph = st.empty()
    st.markdown("---")
    
    st.markdown("### Posture Scores")
    st.write("Good:"); bar_good_ph = st.progress(0)
    st.write("Mild:"); bar_mild_ph = st.progress(0)
    st.write("Severe:"); bar_severe_ph = st.progress(0)
    st.markdown("---")
    
    dist_ph = st.empty() # For deviation
    sound_ph = st.empty()
    tts_ph = st.empty()

    # Update UI (Shared by both Webcam and Upload)
    pred = display_pred
    probs = display_probs

    # 1. Text & Advice
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

    # 2. Progress Bars
    g, m, s = probs.get("good", 0.0)*100, probs.get("mild", 0.0)*100, probs.get("severe", 0.0)*100
    bar_good_ph.progress(int(g), text=f"{g:.1f}%")
    bar_mild_ph.progress(int(m), text=f"{m:.1f}%")
    bar_severe_ph.progress(int(s), text=f"{s:.1f}%")

    # 3. Extra Info
    if ctx and ctx.state.playing:
        dist_ph.markdown(f"Deviation: **{display_dist:.3f}**")
    else:
        dist_ph.empty() # Hide deviation for photo mode

    # 4. Sound Logic (Only for Webcam Loop)
    SOUND_FILE = "alert.mp3"
    if ctx and ctx.state.playing:
        # We need a loop for sound timing, but Streamlit execution model means
        # this part runs once per re-run.
        # However, webrtc context runs in background. 
        # To make sound work reliably in loop, we rely on the main loop below.
        pass

# --- Main Loop for Sound & Continuous Updates (Webcam Only) ---
last_sound_time = 0
SOUND_INTERVAL = 2.0

if ctx and ctx.state.playing:
    while True:
        if not ctx.state.playing: break
        
        # Continuous Update from VideoProcessor
        vp = ctx.video_processor
        if vp:
            probs = vp.latest_probs
            pred = vp.latest_pred
            dist = vp.latest_distance
            
            # Re-update UI placeholders inside loop for real-time effect
            if pred == "good":
                status_ph.markdown("<div class='good-text'>GOOD 😊</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>✅ Perfect! Keep it up.</div>", unsafe_allow_html=True)
            elif pred == "mild":
                status_ph.markdown("<div class='mild-text'>MILD 😐</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>💡 Lift head slightly.<br>Relax shoulders.</div>", unsafe_allow_html=True)
            else:
                status_ph.markdown("<div class='severe-text'>SEVERE 🐢</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>🚨 <b>Pull chin back!</b><br>Open chest.</div>", unsafe_allow_html=True)
            
            # Bars
            g, m, s = probs.get("good", 0.0)*100, probs.get("mild", 0.0)*100, probs.get("severe", 0.0)*100
            bar_good_ph.progress(int(g), text=f"{g:.1f}%")
            bar_mild_ph.progress(int(m), text=f"{m:.1f}%")
            bar_severe_ph.progress(int(s), text=f"{s:.1f}%")
            dist_ph.markdown(f"Deviation: **{dist:.3f}**")

            # Sound
            if pred == "severe":
                current_time = time.time()
                if current_time - last_sound_time > SOUND_INTERVAL:
                    if os.path.exists(SOUND_FILE):
                        sound_html = get_audio_html(SOUND_FILE)
                        sound_ph.markdown(sound_html, unsafe_allow_html=True)
                        last_sound_time = current_time
            else:
                sound_ph.empty()
        
        time.sleep(0.1)
