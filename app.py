import streamlit as st
import cv2
import numpy as np
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from face_detection.predict_emotion import detect_face_emotion, reset_smoothing

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Emotion Detection",
    page_icon="🎭",
    layout="centered"
)

# ─────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ff00cc, #6600ff, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .emotion-box {
        text-align: center;
        padding: 24px 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        margin: 20px 0;
    }
    .emotion-label { font-size: 0.9rem; color: #888; margin-bottom: 6px; }
    .emotion-text  { font-size: 2.6rem; font-weight: 800; color: #00ffcc; }
    .final-box {
        text-align: center;
        padding: 30px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #330033, #1a0033, #000033);
        border: 2px solid #6600ff;
        box-shadow: 0 12px 40px rgba(102,0,255,0.3);
        margin: 24px 0;
    }
    .confidence-bar-container {
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .confidence-label { font-size: 0.8rem; color: #aaa; width: 70px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🎭 Multimodal Emotion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Face · Voice · Text — Fused into one prediction</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio(
    "Select Module:",
    ["🔀 Dashboard (Fused)", "😊 Face Detection", "✍️ Text Detection", "🎤 Voice Detection"]
)

# ─────────────────────────────────────────────
# Emotion Emoji Map
# ─────────────────────────────────────────────
EMOTION_EMOJI = {
    "happy":   "😄",
    "sad":     "😢",
    "angry":   "😡",
    "neutral": "😐",
    "fearful": "😨",
    "fear":    "😨",
    "surprise": "😲",
    "disgust": "🤢",
}

def emotion_emoji(e: str) -> str:
    return EMOTION_EMOJI.get(e.lower(), "🎭")

def render_confidence_bars(scores: dict):
    """Renders horizontal confidence bars for each emotion."""
    if not scores:
        return
    total = sum(scores.values())
    if total == 0:
        return
    normalized = {e: (v / total) * 100 for e, v in scores.items()}
    sorted_scores = sorted(normalized.items(), key=lambda x: -x[1])
    for emotion, pct in sorted_scores:
        emoji = emotion_emoji(emotion)
        st.progress(int(pct), text=f"{emoji} {emotion.capitalize()}: **{pct:.1f}%**")


# ════════════════════════════════════════════════════════
# REAL-TIME VIDEO PROCESSOR (WebRTC)
# ════════════════════════════════════════════════════════

class EmotionVideoProcessor(VideoProcessorBase):
    """
    Processes webcam frames in real-time using streamlit-webrtc.
    Runs DeepFace every N frames to keep latency low,
    and overlays the emotion label on every frame.
    """
    ANALYZE_EVERY_N = 8  # Analyze 1 out of every 8 frames for performance

    def __init__(self):
        self._frame_count = 0
        self._last_emotion = "neutral"
        self._last_scores = {}
        self._lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        # Only run DeepFace every N frames
        if self._frame_count % self.ANALYZE_EVERY_N == 0:
            try:
                annotated, emotion, scores = detect_face_emotion(img.copy(), use_smoothing=True)
                with self._lock:
                    self._last_emotion = emotion
                    self._last_scores = scores
                    img = annotated
            except Exception:
                pass
        else:
            # Draw cached emotion on current frame
            with self._lock:
                emotion = self._last_emotion
                scores = self._last_scores

            # Overlay emotion label at top-left
            EMOTION_COLORS = {
                "happy":    (0, 220, 100),
                "sad":      (255, 100, 50),
                "angry":    (0, 60, 255),
                "neutral":  (200, 200, 200),
                "fearful":  (30, 165, 255),
                "surprise": (0, 230, 230),
                "disgust":  (160, 30, 220),
            }
            color = EMOTION_COLORS.get(emotion, (0, 255, 0))
            label = f"{emotion.upper()}"
            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            cv2.rectangle(img, (10, 10), (20 + tw, 45 + th), (0, 0, 0), -1)
            cv2.putText(img, label, (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest(self):
        with self._lock:
            return self._last_emotion, dict(self._last_scores)


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ════════════════════════════════════════════════════════
# FACE DETECTION MODULE
# ════════════════════════════════════════════════════════
if app_mode == "😊 Face Detection":
    st.header("😊 Real-time Face Emotion Detection")

    tab_live, tab_photo = st.tabs(["🎥 Live Camera", "📸 Single Photo"])

    # ── Tab 1: Live Real-time ────────────────────────────
    with tab_live:
        st.write("Your webcam feed is analyzed live — emotion updates every few frames.")

        ctx = webrtc_streamer(
            key="face-emotion-live",
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Live emotion display below the video
        emotion_placeholder = st.empty()
        scores_placeholder  = st.empty()

        if ctx.state.playing and ctx.video_processor:
            import time
            while ctx.state.playing:
                emotion, scores = ctx.video_processor.get_latest()
                emotion_placeholder.markdown(f'''
                <div class="emotion-box" style="margin-top:12px;">
                    <div class="emotion-label">Live Detected Emotion</div>
                    <div class="emotion-text">{emotion_emoji(emotion)} {emotion.upper()}</div>
                </div>
                ''', unsafe_allow_html=True)
                with scores_placeholder.container():
                    if scores:
                        st.caption("Confidence Breakdown")
                        render_confidence_bars(scores)
                time.sleep(0.3)  # Refresh UI 3× per second

    # ── Tab 2: Single Photo ──────────────────────────────
    with tab_photo:
        st.write("Take a single snapshot and analyze it.")
        reset_smoothing()
        face_img = st.camera_input("Take a picture")

        if face_img is not None:
            from PIL import Image
            image = Image.open(face_img)
            img_array = np.array(image)

            with st.spinner("Analyzing face with DeepFace..."):
                annotated_frame, emotion, scores = detect_face_emotion(img_array, use_smoothing=False)

            st.image(annotated_frame, channels="RGB", use_container_width=True)
            st.markdown(f'''
            <div class="emotion-box">
                <div class="emotion-label">Detected Emotion</div>
                <div class="emotion-text">{emotion_emoji(emotion)} {emotion.upper()}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.subheader("Confidence Breakdown")
            render_confidence_bars(scores)


# ════════════════════════════════════════════════════════
# TEXT DETECTION MODULE
# ════════════════════════════════════════════════════════
elif app_mode == "✍️ Text Detection":
    st.header("✍️ Text Emotion Detection")
    st.write("Enter text to analyze its emotional content using keyword scoring + VADER NLP.")

    from text_detection.predict_emotion import predict_text_emotion

    user_text = st.text_area("Enter some text here:", "I feel amazing today!", height=120)

    if st.button("🔍 Analyze Text", type="primary"):
        if user_text.strip():
            with st.spinner("Analyzing text..."):
                emotion = predict_text_emotion(user_text)

            if "Error" in emotion:
                st.error(emotion)
            else:
                st.markdown(f'''
                <div class="emotion-box">
                    <div class="emotion-label">Detected Emotion</div>
                    <div class="emotion-text">{emotion_emoji(emotion)} {emotion.upper()}</div>
                </div>
                ''', unsafe_allow_html=True)

            # Show quick test examples
            with st.expander("Try more examples"):
                examples = [
                    "I am so happy today!",
                    "This is making me so angry!",
                    "I feel really sad and alone.",
                    "Oh wow, I didn't expect that at all!",
                    "I'm terrified about what might happen.",
                    "It's just a regular Tuesday.",
                ]
                for ex in examples:
                    em = predict_text_emotion(ex)
                    st.write(f"{emotion_emoji(em)} **{em.upper()}** — _{ex}_")
        else:
            st.warning("Please enter some text to analyze.")


# ════════════════════════════════════════════════════════
# VOICE DETECTION MODULE
# ════════════════════════════════════════════════════════
elif app_mode == "🎤 Voice Detection":
    st.header("🎤 Voice Emotion Detection")
    st.write("Upload a `.wav` file to analyze speech emotion from 6 acoustic features.")

    from voice_detection.predict_emotion import predict_voice_emotion_with_scores

    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("🔊 Analyze Audio", type="primary"):
            tmp_path = "temp_uploaded.wav"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Extracting acoustic features..."):
                emotion, scores = predict_voice_emotion_with_scores(tmp_path)

            if "Error" in emotion:
                st.error(emotion)
            else:
                st.markdown(f'''
                <div class="emotion-box">
                    <div class="emotion-label">Detected Voice Emotion</div>
                    <div class="emotion-text">{emotion_emoji(emotion)} {emotion.upper()}</div>
                </div>
                ''', unsafe_allow_html=True)

                st.subheader("Confidence Breakdown")
                render_confidence_bars(scores)

            # Info about features used
            with st.expander("ℹ️ How voice analysis works"):
                st.markdown("""
                The voice module extracts **6 acoustic features** per recording:
                | Feature | What it captures |
                |---|---|
                | **RMS Energy** | Volume / Loudness |
                | **Pitch (F0)** | Fundamental frequency |
                | **Zero Crossing Rate** | Breathiness / fricative sounds |
                | **Spectral Centroid** | Brightness / sharpness of sound |
                | **MFCC Variance** | Expressiveness / vocal variability |
                | **Onset Strength** | Speaking rate / energy bursts |

                Each feature is normalized and scored against emotion profiles.
                """)


# ════════════════════════════════════════════════════════
# DASHBOARD — WEIGHTED FUSION
# ════════════════════════════════════════════════════════
elif app_mode == "🔀 Dashboard (Fused)":
    st.header("📊 Multimodal Emotion Fusion")
    st.write("Provide inputs from **Face**, **Voice**, and **Text** to get a weighted combined prediction.")

    # Fusion weights (Face is most reliable via DeepFace, then Voice, then Text)
    WEIGHTS = {"Face": 0.45, "Voice": 0.35, "Text": 0.20}

    from text_detection.predict_emotion import predict_text_emotion
    from voice_detection.predict_emotion import predict_voice_emotion_with_scores
    from PIL import Image

    CANONICAL_EMOTIONS = ["happy", "sad", "angry", "neutral", "fearful", "surprise", "disgust"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Fusion Weights")
    face_w  = st.sidebar.slider("Face weight",  0.0, 1.0, 0.45, 0.05)
    voice_w = st.sidebar.slider("Voice weight", 0.0, 1.0, 0.35, 0.05)
    text_w  = st.sidebar.slider("Text weight",  0.0, 1.0, 0.20, 0.05)

    total_w = face_w + voice_w + text_w
    if total_w == 0:
        total_w = 1.0
    WEIGHTS = {
        "Face":  face_w / total_w,
        "Voice": voice_w / total_w,
        "Text":  text_w / total_w
    }

    # ── Input sections ───────────────────────────────────────────
    st.subheader("1️⃣ Text Input")
    user_text = st.text_area("Enter your thoughts:", key="multi_text", height=80)

    st.subheader("2️⃣ Face Input")
    use_camera = st.checkbox("📷 Turn Camera On")
    face_img = None
    if use_camera:
        reset_smoothing()
        face_img = st.camera_input("Take a picture to capture your facial expression")
    else:
        st.info("Camera is off. Check the box above to turn it on.")

    st.subheader("3️⃣ Voice Input")
    voice_file = st.file_uploader("Upload a voice recording (.wav)", type=['wav'], key="multi_voice")

    # ── Fuse ─────────────────────────────────────────────────────
    if st.button("🚀 Predict Combined Emotion", type="primary"):
        with st.spinner("Running multimodal analysis..."):

            fused_scores = {e: 0.0 for e in CANONICAL_EMOTIONS}
            results = {}
            individual_scores = {}
            modalities_used = []

            # ─── Text ───────────────────────────────────────────
            if user_text.strip():
                text_em = predict_text_emotion(user_text)
                if "Error" not in text_em:
                    results["Text"] = text_em
                    modalities_used.append("Text")
                    # Simple one-hot style: winning emotion gets full weight
                    for e in CANONICAL_EMOTIONS:
                        if e == text_em.lower():
                            fused_scores[e] += WEIGHTS["Text"] * 1.0
                    individual_scores["Text"] = {e: (100.0 if e == text_em.lower() else 0.0) for e in CANONICAL_EMOTIONS}
                else:
                    results["Text"] = "❌ Error"
            else:
                results["Text"] = "⚪ No Input"

            # ─── Face ───────────────────────────────────────────
            if face_img is not None:
                image = Image.open(face_img)
                img_array = np.array(image)
                _, face_em, face_scores = detect_face_emotion(img_array, use_smoothing=False)
                if face_em:
                    results["Face"] = face_em
                    modalities_used.append("Face")
                    total_face = sum(face_scores.values()) or 1
                    for e in CANONICAL_EMOTIONS:
                        fused_scores[e] += WEIGHTS["Face"] * (face_scores.get(e, 0.0) / total_face)
                    individual_scores["Face"] = {e: face_scores.get(e, 0.0) for e in CANONICAL_EMOTIONS}
                else:
                    results["Face"] = "⚪ Not Detected"
            else:
                results["Face"] = "⚪ No Image"

            # ─── Voice ──────────────────────────────────────────
            if voice_file is not None:
                with open("temp_multi.wav", "wb") as f:
                    f.write(voice_file.getbuffer())
                voice_em, voice_scores = predict_voice_emotion_with_scores("temp_multi.wav")
                if "Error" not in voice_em:
                    results["Voice"] = voice_em
                    modalities_used.append("Voice")
                    total_voice = sum(voice_scores.values()) or 1
                    for e in CANONICAL_EMOTIONS:
                        fused_scores[e] += WEIGHTS["Voice"] * (voice_scores.get(e, 0.0) / total_voice)
                    individual_scores["Voice"] = {e: voice_scores.get(e, 0.0) for e in CANONICAL_EMOTIONS}
                else:
                    results["Voice"] = "❌ Error"
            else:
                results["Voice"] = "⚪ No Audio"

            # ─── Display Results ─────────────────────────────────
            st.divider()
            st.subheader("📋 Individual Results")
            col1, col2, col3 = st.columns(3)
            for col, (modality, icon) in zip([col1, col2, col3],
                                             [("Face", "😊"), ("Text", "✍️"), ("Voice", "🎤")]):
                with col:
                    em = results.get(modality, "⚪ N/A")
                    display = f"{emotion_emoji(em)} {em.upper()}" if em not in ["⚪ No Input", "⚪ No Image", "⚪ No Audio", "⚪ Not Detected", "❌ Error"] else em
                    st.metric(label=f"{icon} {modality}", value=display)
                    if modality in individual_scores:
                        with st.expander("Scores"):
                            render_confidence_bars(individual_scores[modality])

            # ─── Final Fusion Decision ───────────────────────────
            st.divider()
            st.subheader("🏆 Final Decision")

            if len(modalities_used) > 0:
                # Re-normalize fused scores by number of active modalities
                active_weight = sum(WEIGHTS[m] for m in modalities_used)
                if active_weight > 0:
                    fused_scores = {e: v / active_weight for e, v in fused_scores.items()}

                final_emotion = max(fused_scores, key=fused_scores.get)
                final_confidence = fused_scores[final_emotion] * 100

                st.markdown(f'''
                <div class="final-box">
                    <div style="font-size:0.9rem; color:#aaa; margin-bottom:8px;">
                        Fusion: Face×{WEIGHTS["Face"]:.0%} + Voice×{WEIGHTS["Voice"]:.0%} + Text×{WEIGHTS["Text"]:.0%}
                    </div>
                    <div style="font-size:4rem;">{emotion_emoji(final_emotion)}</div>
                    <div style="font-size:2.8rem; font-weight:900; color:#ff00cc;">{final_emotion.upper()}</div>
                    <div style="font-size:1rem; color:#aaa; margin-top:8px;">
                        Confidence: <strong style="color:#00ffcc">{final_confidence:.1f}%</strong>
                        &nbsp;·&nbsp; Based on: <strong>{", ".join(modalities_used)}</strong>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.subheader("Fused Score Breakdown")
                render_confidence_bars(fused_scores)

            else:
                st.warning("Please provide at least one input (text, face, or voice) to get a prediction.")
