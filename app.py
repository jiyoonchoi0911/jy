import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Page Configuration ---
st.set_page_config(page_title="AI Real-time Posture Correction", page_icon="🐢")

# 스타일 설정
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold; font-size: 20px;}
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 20px;}
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 20px;}
    .warning-box { background-color: #fadbd8; border: 2px solid #e74c3c; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐢 AI Real-time Turtle Neck Diagnosis")
st.write("Turn on the webcam to analyze your posture in real-time.")

# --- Load Model & MediaPipe ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()
mp_pose = mp.solutions.pose

# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model
        # 결과 공유를 위한 변수
        self.latest_probs = {'good': 0, 'mild': 0, 'severe': 0}
        self.latest_pred = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # 2. Feature Extraction
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1

                indices = [0, 2, 5, 7, 8, 11, 12]
                features = []
                
                h, w, _ = img.shape
                draw_points = []

                for idx in indices:
                    lm = landmarks[idx]
                    norm_x = (lm.x - center_x) / width
                    norm_y = (lm.y - center_y) / width
                    features.extend([norm_x, norm_y])
                    draw_points.append((int(lm.x * w), int(lm.y * h)))

                # 3. Prediction
                if self.model:
                    probs = self.model.predict_proba([features])[0]
                    classes = self.model.classes_
                    
                    # 공유 변수 업데이트
                    self.latest_probs = {cls: p for cls, p in zip(classes, probs)}
                    self.latest_pred = self.model.predict([features])[0]
                    
                    # 4. 화면에는 점만 찍기 (텍스트 없음)
                    for px, py in draw_points:
                        cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                    
            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main Tab Configuration ---
tab1, tab2 = st.tabs(["📷 Real-time Analysis", "🖼️ Upload Photo"])

# Tab 1: Real-time with External UI
with tab1:
    st.header("Real-time Webcam")
    
    if model is None:
        st.error("Model file (posture_model.pkl) is missing.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # WebRTC Streamer
            ctx = webrtc_streamer(
                key="posture-check",
                video_processor_factory=VideoProcessor,
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True
            )

        with col2:
            st.subheader("Live Status")
            status_text_ph = st.empty()
            
            st.write("**Prediction Confidence:**")
            bar_good_ph = st.empty()
            bar_mild_ph = st.empty()
            bar_severe_ph = st.empty()
            warning_ph = st.empty()

        if ctx.state.playing:
            while True:
                if ctx.video_processor:
                    probs = ctx.video_processor.latest_probs
                    pred = ctx.video_processor.latest_pred
                    
                    if pred:
                        p_good = int(probs.get('good', 0) * 100)
                        p_mild = int(probs.get('mild', 0) * 100)
                        p_severe = int(probs.get('severe', 0) * 100)

                        if pred == 'good':
                            status_text_ph.markdown(f"<p class='good-text'>Status: GOOD 😊</p>", unsafe_allow_html=True)
                        elif pred == 'mild':
                            status_text_ph.markdown(f"<p class='mild-text'>Status: MILD 😐</p>", unsafe_allow_html=True)
                        else:
                            status_text_ph.markdown(f"<p class='severe-text'>Status: SEVERE 🐢</p>", unsafe_allow_html=True)

                        bar_good_ph.progress(p_good, text=f"Good: {p_good}%")
                        bar_mild_ph.progress(p_mild, text=f"Mild: {p_mild}%")
                        bar_severe_ph.progress(p_severe, text=f"Severe: {p_severe}%")

                        if pred == 'severe':
                            warning_ph.markdown("""
                                <div class='warning-box'>
                                    🚨 <b>BAD POSTURE DETECTED!</b><br>
                                    Please straighten your neck.
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            warning_ph.empty()
                    
                import time
                time.sleep(0.1)

# Tab 2: Upload
with tab2:
    st.header("File Upload Diagnosis")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_np = np.array(image.convert('RGB'))
        pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
        results = pose_static.process(img_np)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1
                
                features = []
                indices = [0, 2, 5, 7, 8, 11, 12]
                for idx in indices:
                    lm = landmarks[idx]
                    features.extend([(lm.x - center_x)/width, (lm.y - center_y)/width])
                
                probs = model.predict_proba([features])[0]
                classes = model.classes_
                prob_dict = {cls: round(p * 100, 1) for cls, p in zip(classes, probs)}
                
                st.subheader("Analysis Result")
                st.write(f"**Good: {prob_dict.get('good', 0)}%**")
                st.progress(int(prob_dict.get('good', 0)))
                st.write(f"**Mild: {prob_dict.get('mild', 0)}%**")
                st.progress(int(prob_dict.get('mild', 0)))
                st.write(f"**Severe: {prob_dict.get('severe', 0)}%**")
                st.progress(int(prob_dict.get('severe', 0)))
                
                pred = model.predict([features])[0]
                if pred == 'severe':
                    st.error("🚨 WARNING: Severe Forward Head Posture detected!")
                elif pred == 'mild':
                    st.warning("🟡 Caution: Mild Forward Head Posture.")
                else:
                    st.success("🟢 Good Posture!")
            except:
                st.error("Analysis failed.")
        else:
            st.error("Person not found.")
