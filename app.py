import streamlit as st, cv2, numpy as np, mediapipe as mp, av, time, base64, os, joblib, gc
from PIL import Image
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. 메모리 절약을 위한 가비지 컬렉션 강제 실행
gc.collect()

st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

@st.cache_resource
def load_model():
    try: return joblib.load('posture_model.pkl')
    except: return None
model = load_model()

def get_audio_html(f):
    if not os.path.exists(f): return ""
    with open(f, "rb") as o: b64 = base64.b64encode(o.read()).decode()
    return f'<audio autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio><div style="display:none;">{time.time()}</div>'

st.markdown("""<script>
window.lp=null;function sp(s){if(!('speechSynthesis' in window))return;var t='';if(s==='GOOD')t='Posture is good.';else if(s==='MILD')t='Posture is mild.';else if(s==='SEVERE')t='Posture is severe.';if(t==='')return;var u=new SpeechSynthesisUtterance(t);u.lang='en-US';window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}
function updatePostureStatus(s){if(window.lp===s)return;window.lp=s;sp(s);}
</script><style>.good-text{color:#2ecc71;font-weight:bold;font-size:40px;text-align:center}.mild-text{color:#f1c40f;font-weight:bold;font-size:40px;text-align:center}.severe-text{color:#e74c3c;font-weight:bold;font-size:40px;text-align:center;animation:blink 1s infinite}.advice-box{background-color:#fff9c4;padding:15px;border-radius:10px;border-left:5px solid #fbc02d;font-size:18px;font-weight:bold;color:#333;margin:10px 0 20px 0}@keyframes blink{50%{opacity:0.5}}.stProgress>div>div>div>div{background-color:#2ecc71}</style>""", unsafe_allow_html=True)

st.title("🐢 AI Posture Correction Pro")
st.markdown("**Webcam:** Sit straight & Click 'Set Standard'. **Upload:** Auto-diagnosis using AI.")
mp_pose = mp.solutions.pose

def get_probs(d, tg=0.12, tm=0.28):
    d=float(d); g=max(0.0,1.0-d/tg) if d<=tg else 0.0; m=d/tg if d<=tg else (1.0-(d-tg)/(tm-tg) if d<=tm else 0.0); s=0.0 if d<=tm else min(1.0,(d-tm)/tm)
    tot=g+m+s; return {"good":1/3,"mild":1/3,"severe":1/3} if tot==0 else {"good":g/tot,"mild":m/tot,"severe":s/tot}

def get_fts(lms, shape):
    h,w,_=shape; ls,rs=lms[11],lms[12]; cx,cy=(ls.x+rs.x)/2,(ls.y+rs.y)/2
    wd=np.linalg.norm([ls.x-rs.x, ls.y-rs.y]) or 1; fts=[]; kps={}
    for i in [0,2,5,7,8,11,12]: lm=lms[i]; fts.extend([(lm.x-cx)/wd,(lm.y-cy)/wd]); kps[i]=(int(lm.x*w),int(lm.y*h))
    return fts, kps

def draw(img, kps, p):
    c=(0,255,0) if p=="good" else (0,255,255) if p=="mild" else (0,0,255)
    for v in kps.values(): cv2.circle(img,v,5,c,-1)
    if 11 in kps and 12 in kps: cv2.line(img,kps[11],kps[12],c,2)
    if 0 in kps and 11 in kps: cv2.line(img,((kps[11][0]+kps[12][0])//2,(kps[11][1]+kps[12][1])//2),kps[0],c,2)
    return img

class VP(VideoTransformerBase):
    def __init__(self):
        # 2. 모델 복잡도를 0(Lite)으로 설정하여 메모리 절약 (기본값 1 -> 0)
        self.pose=mp_pose.Pose(min_detection_confidence=0.5, model_complexity=0)
        self.base=None; self.cal=False; self.hist=deque(maxlen=10)
        self.res={"good":0,"mild":0,"severe":0}; self.pred="good"; self.dist=0
    def recv(self, f):
        img=f.to_ndarray(format="bgr24"); res=self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            fts, kps = get_fts(res.pose_landmarks.landmark, img.shape)
            if self.cal: self.base=np.array(fts); self.hist.clear(); self.cal=False
            if self.base is not None:
                self.hist.append(np.linalg.norm(np.array(fts)-self.base)); self.dist=np.mean(self.hist)
                self.res=get_probs(self.dist); self.pred=max(self.res, key=self.res.get)
            else: self.pred="good"
            draw(img, kps, self.pred)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

c1, c2 = st.columns([3,2])
probs, pred, dist = {"good":0,"mild":0,"severe":0}, "good", 0

with c1:
    t1, t2 = st.tabs(["📹 Live Webcam", "🖼️ Upload Photo"])
    with t1:
        top = st.container()
        # 3. 비디오 해상도를 낮춰 메모리 사용량 감소 (width: ideal 480)
        ctx = webrtc_streamer(
            key="bp", 
            video_processor_factory=VP, 
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}, 
            media_stream_constraints={"video":{"width":{"ideal":480}},"audio":False}, 
            async_processing=True
        )
        with top:
            if st.button("📏 Set Current Posture as Standard", type="primary", use_container_width=True):
                if ctx and ctx.video_processor: ctx.video_processor.cal=True; st.success("✅ Set!")
                else: st.warning("Start cam first.")
            st.markdown("---")
    with t2:
        up = st.file_uploader("Upload", type=['jpg','png','jpeg'])
        if up and model:
            img = np.array(Image.open(up).convert('RGB'))
            with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
                res = pose.process(img)
                if res.pose_landmarks:
                    fts, kps = get_fts(res.pose_landmarks.landmark, img.shape)
                    probs = dict(zip(model.classes_, model.predict_proba([fts])[0]))
                    pred = model.predict([fts])[0]
                    st.image(draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), kps, pred), caption="Result", use_column_width=True)
                else: st.error("No person detected.")

if ctx and ctx.state.playing and ctx.video_processor:
    probs, pred, dist = ctx.video_processor.res, ctx.video_processor.pred, ctx.video_processor.dist

with c2:
    st.markdown("### 📊 Status Report")
    st_ph = st.empty(); adv_ph = st.empty(); js_ph = st.empty(); st.markdown("---"); st.markdown("### Scores")
    bar_ph = {k: st.progress(0) for k in ["good","mild","severe"]}; st.markdown("---"); d_ph = st.empty(); snd_ph = st.empty()
    
    def ui(p, pr, d):
        st_ph.markdown(f"<div class='{p}-text'>{p.upper()} {'😊' if p=='good' else '😐' if p=='mild' else '🐢'}</div>", unsafe_allow_html=True)
        msg = "✅ Perfect!" if p=="good" else "💡 Lift head." if p=="mild" else "🚨 <b>Pull chin back!</b>"
        adv_ph.markdown(f"<div class='advice-box'>{msg}</div>", unsafe_allow_html=True)
        js_ph.markdown(f"<script>updatePostureStatus('{p.upper()}');</script>", unsafe_allow_html=True)
        for k, v in pr.items(): bar_ph[k].progress(int(v*100), text=f"{k.capitalize()}: {v*100:.1f}%")
        d_ph.markdown(f"Deviation: **{d:.3f}**") if ctx and ctx.state.playing else d_ph.empty()

    ui(pred, probs, dist)
    
    last_snd = 0
    if ctx and ctx.state.playing:
        while True:
            if not ctx.state.playing: break
            vp = ctx.video_processor
            if vp:
                ui(vp.pred, vp.res, vp.dist)
                if vp.pred=="severe" and time.time()-last_snd > 2.0:
                    if os.path.exists("alert.mp3"): snd_ph.markdown(get_audio_html("alert.mp3"), unsafe_allow_html=True)
                    last_snd = time.time()
                elif vp.pred!="severe": snd_ph.empty()
            time.sleep(0.1)
