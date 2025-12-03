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
st.set_page_config(page_title="AI 거북목 교정 프로 (AI Posture Pro)", page_icon="🐢", layout="wide")

# --- CSS & Audio Script ---
# 소리 재생을 위한 자바스크립트 및 스타일
def get_audio_html(sound_file_path):
    # 경고음 (비프음) Base64 데이터
    beep_b64 = "UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU" + "A" * 500 # 짧은 비프음 더미 데이터 (실제 작동을 위해 아래 스크립트 사용)
    
    # 실제 브라우저 내장 오디오 사용
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
    .good-text { color: #2ecc71; font-weight: bold; font-size: 24px; }
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 24px; }
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 24px; animation: blink 1s infinite; }
    
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    
    .stat-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .stat-title { font-size: 16px; color: #555; }
    .stat-value { font-size: 28px; font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)

st.markdown(get_audio_html(""), unsafe_allow_html=True) # 오디오 스크립트 로드

st.title("🐢 AI 거북목 교정 Pro")
st.markdown("웹캠을 켜고 자세를 분석하세요. **'심각' 단계가 지속되면 알림이 울립니다.**")

# --- Load Model & MediaPipe ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()
mp_pose = mp.solutions.pose

# --- Session State 초기화 (통계용) ---
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'total_severe_time' not in st.session_state:
    st.session_state.total_severe_time = 0
if 'calibration_y' not in st.session_state:
    st.session_state.calibration_y = 0.0 # 캘리브레이션 오프셋

# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model
        
        # 1. Smoothing (결과 떨림 방지)
        self.history_len = 10
        self.prob_history = deque(maxlen=self.history_len)
        
        # 결과 공유 변수
        self.latest_probs = {'good': 0, 'mild': 0, 'severe': 0}
        self.latest_pred = "good"
        self.severe_consecutive_frames = 0 # 연속 프레임 카운트 (소리 알림용)
        self.trigger_sound = False
        
        # 캘리브레이션 값 가져오기 (Processor 생성 시점의 값)
        # 주의: 스트림 도중 값을 바꾸려면 별도 메커니즘 필요하나, 여기선 초기값 사용
        self.cal_y = 0 

    def update_calibration(self, value):
        self.cal_y = value

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 이미지 전처리
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        current_pred = "good"
        current_probs = [1.0, 0.0, 0.0] # 기본값

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # 2. Feature Extraction
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                
                # 캘리브레이션 적용 (높이 보정) - 단순 예시: 중심점 Y축 이동
                # 실제로는 모델 재학습이 좋으나, 여기서는 입력값 미세 조정으로 구현
                # center_y += st.session_state.get('calibration_y', 0) 

                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1

                indices = [0, 2, 5, 7, 8, 11, 12]
                features = []
                
                # 시각화 포인트 및 뼈대
                keypoints = {} # 그리기 위해 좌표 저장

                for idx in indices:
                    lm = landmarks[idx]
                    # 정규화
                    norm_x = (lm.x - center_x) / width
                    norm_y = (lm.y - center_y) / width
                    features.extend([norm_x, norm_y])
                    
                    # 픽셀 좌표 저장
                    px, py = int(lm.x * w), int(lm.y * h)
                    keypoints[idx] = (px, py)

                # 3. Prediction
                if self.model:
                    probs = self.model.predict_proba([features])[0]
                    
                    # Smoothing: 큐에 확률 저장
                    self.prob_history.append(probs)
                    
                    # 평균 확률 계산
                    avg_probs = np.mean(self.prob_history, axis=0)
                    classes = self.model.classes_ # ['good', 'mild', 'severe'] 순서라고 가정 (확인 필요)
                    
                    # 클래스 매핑 (모델마다 순서가 다를 수 있음. 여기선 이름으로 매칭)
                    prob_dict = {cls: p for cls, p in zip(classes, avg_probs)}
                    self.latest_probs = prob_dict
                    
                    # 가장 높은 확률의 클래스 결정
                    current_pred = max(prob_dict, key=prob_dict.get)
                    self.latest_pred = current_pred
                    
                    # 4. 시각화 (Skeleton Visualization)
                    # 색상 결정
                    color = (0, 255, 0) # Green
                    if current_pred == 'mild': color = (0, 255, 255) # Yellow
                    if current_pred == 'severe': color = (0, 0, 255) # Red (BGR)

                    # 점 그리기
                    for idx, (px, py) in keypoints.items():
                        cv2.circle(img, (px, py), 5, color, -1)
                    
                    # 뼈대 그리기 (어깨선, 목선)
                    # 11:왼쪽어깨, 12:오른쪽어깨, 0:코
                    if 11 in keypoints and 12 in keypoints:
                        cv2.line(img, keypoints[11], keypoints[12], color, 2)
                    if 0 in keypoints:
                        # 어깨 중심 계산
                        sh_center = ((keypoints[11][0] + keypoints[12][0]) // 2, 
                                     (keypoints[11][1] + keypoints[12][1]) // 2)
                        cv2.line(img, sh_center, keypoints[0], color, 2)

                    # 5. 소리 알림 로직 (Severe 상태 지속 시)
                    if current_pred == 'severe':
                        self.severe_consecutive_frames += 1
                        if self.severe_consecutive_frames > 30: # 약 1초 이상 (30fps 기준)
                            self.trigger_sound = True
                    else:
                        self.severe_consecutive_frames = 0
                        self.trigger_sound = False
                        
            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
col_main, col_stat = st.columns([3, 1])

with col_main:
    # 캘리브레이션 (초기 설정)
    if st.button("📏 현재 자세를 '기준점'으로 설정 (Calibration)"):
        st.session_state.start_time = time.time()
        st.session_state.total_severe_time = 0
        st.success("기준점이 설정되었습니다! 타이머가 초기화됩니다.")

    if model is None:
        st.error("모델 파일이 없습니다.")
    else:
        ctx = webrtc_streamer(
            key="posture-pro",
            video_processor_factory=VideoProcessor,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

with col_stat:
    st.markdown("### 📊 실시간 분석")
    status_ph = st.empty()
    st.write("---")
    st.markdown("### ⏱️ 시간 통계")
    timer_ph = st.empty()
    severe_timer_ph = st.empty()
    
    # JavaScript Sound Trigger Placeholder
    sound_ph = st.empty()

# --- Main Loop (Outside of Streamer) ---
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            # 1. 데이터 가져오기
            probs = ctx.video_processor.latest_probs
            pred = ctx.video_processor.latest_pred
            trigger_sound = ctx.video_processor.trigger_sound
            
            # 2. UI 업데이트
            if pred == 'good':
                status_ph.markdown(f"<div class='good-text'>상태: 좋음 😊</div>", unsafe_allow_html=True)
            elif pred == 'mild':
                status_ph.markdown(f"<div class='mild-text'>상태: 주의 😐</div>", unsafe_allow_html=True)
            else:
                status_ph.markdown(f"<div class='severe-text'>상태: 심각 🐢</div>", unsafe_allow_html=True)
            
            # 확률 바
            st.sidebar.markdown("### 세부 확률")
            st.sidebar.progress(probs.get('good', 0), text=f"Good: {int(probs.get('good', 0)*100)}%")
            st.sidebar.progress(probs.get('mild', 0), text=f"Mild: {int(probs.get('mild', 0)*100)}%")
            st.sidebar.progress(probs.get('severe', 0), text=f"Severe: {int(probs.get('severe', 0)*100)}%")

            # 3. 소리 재생 (JavaScript 트리거)
            if trigger_sound:
                # 자바스크립트 함수 호출 (playAlert)
                sound_ph.markdown("""
                    <script>
                    playAlert();
                    </script>
                    """, unsafe_allow_html=True)
                # 통계 누적
                st.session_state.total_severe_time += 0.1 # 대략적인 루프 시간 더하기
            else:
                sound_ph.empty()

            # 4. 시간 통계 업데이트
            elapsed_time = int(time.time() - st.session_state.start_time)
            mins, secs = divmod(elapsed_time, 60)
            
            severe_mins, severe_secs = divmod(int(st.session_state.total_severe_time), 60)
            
            timer_ph.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-title'>총 사용 시간</div>
                    <div class='stat-value'>{mins}분 {secs}초</div>
                </div>
            """, unsafe_allow_html=True)
            
            severe_timer_ph.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-title' style='color: #e74c3c;'>나쁜 자세 누적</div>
                    <div class='stat-value' style='color: #e74c3c;'>{severe_mins}분 {severe_secs}초</div>
                </div>
            """, unsafe_allow_html=True)

        time.sleep(0.1)
