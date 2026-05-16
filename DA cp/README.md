# 🎭 Multimodal Emotion Detection System

A real-time emotion detection system that fuses **Face**, **Voice**, and **Text** inputs into a single combined emotion prediction using a weighted multimodal fusion engine.

Built with Python, Streamlit, DeepFace, Librosa, and NLTK.

---

## ✨ Features

| Module | Technology | Emotions Detected |
|---|---|---|
| 😊 **Face (Live)** | DeepFace + WebRTC | happy, sad, angry, neutral, fearful, surprise, disgust |
| 🎤 **Voice** | Librosa (6 acoustic features) | happy, sad, angry, neutral, fearful, surprise |
| ✍️ **Text** | NLTK VADER + Keyword Lexicon | happy, sad, angry, neutral, fearful, surprise, disgust |
| 🔀 **Fusion** | Weighted voting (Face 45%, Voice 35%, Text 20%) | All of the above |

### 🔍 How Each Module Works

**Face Detection**
- Uses `streamlit-webrtc` for real-time webcam streaming
- DeepFace analyzes frames every 8th frame for smooth performance
- Temporal smoothing (5-frame rolling average) reduces prediction jitter
- Full confidence score breakdown displayed per emotion

**Voice Detection**
- Extracts 6 acoustic features: RMS Energy, Pitch (F0), Zero Crossing Rate, Spectral Centroid, MFCC Variance, Onset Strength
- Features normalized to [0,1] per sample — works across all microphones
- Scored against emotion profiles using a range-fit metric

**Text Detection**
- Keyword lexicon with 30+ words per emotion category
- Negation handling ("not happy" → sad)
- Intensity boosters for ALL CAPS, `!!!`, repeated letters
- VADER sentiment used as a secondary scoring booster

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-emotion-detection.git
cd multimodal-emotion-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
.
├── app.py                        # Main Streamlit application
├── requirements.txt              # All Python dependencies
│
├── face_detection/
│   └── predict_emotion.py        # DeepFace + temporal smoothing
│
├── text_detection/
│   ├── predict_emotion.py        # Keyword lexicon + VADER NLP
│   └── train_text_model.py       # (Optional) Naive Bayes trainer
│
├── voice_detection/
│   ├── predict_emotion.py        # Acoustic feature scoring
│   └── train_voice_model.py      # (Optional) Random Forest trainer
│
└── datasets/                     # (Not included — see below)
```

---

## 📦 Dependencies

Key packages used:
- `streamlit` — Web UI framework
- `streamlit-webrtc` + `av` — Real-time webcam streaming
- `deepface` — Face emotion recognition (pre-trained CNNs)
- `librosa` — Audio feature extraction
- `nltk` — NLP (VADER sentiment analysis)
- `opencv-python-headless` — Image processing
- `scikit-learn` — ML utilities

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📊 Optional: Training Your Own Models

The app uses heuristic engines by default (no dataset needed).  
If you want to train ML models on real datasets:

**Voice** — Download [RAVDESS](https://zenodo.org/record/1188976) and place in `datasets/ravdess/`:
```bash
python voice_detection/train_voice_model.py
```

**Text** — Uses built-in sample data:
```bash
python text_detection/train_text_model.py
```

---

## 🧠 Multimodal Fusion

The Dashboard tab combines all three modalities using **weighted scoring**:

```
Final Score = Face×0.45 + Voice×0.35 + Text×0.20
```

Weights are **adjustable via sliders** in the sidebar. The emotion with the highest fused score wins.

---

## 📸 Screenshots

> Add your own screenshots here after running the app.

---

## 📄 License

MIT License — free to use, modify, and distribute.
