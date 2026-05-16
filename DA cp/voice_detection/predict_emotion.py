# pyrefly: ignore [missing-import]
import librosa
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# Emotion profile table (weights for each acoustic feature per emotion)
#
# Features:  rms  pitch  zcr  centroid  mfcc_var  onset
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is a tuple of ideal (normalized) feature ranges:
#   (rms_low, rms_high, pitch_low, pitch_high, zcr_low, zcr_high,
#    centroid_low, centroid_high, mfcc_var_low, mfcc_var_high,
#    onset_low, onset_high)
#
# The scoring engine computes how well a sample fits each profile.

EMOTION_PROFILES = {
    # High energy, high pitch, high spectral brightness, high ZCR
    "angry":   dict(rms=(0.55, 1.0), pitch=(0.5, 1.0), zcr=(0.5, 1.0),
                    centroid=(0.55, 1.0), mfcc_var=(0.5, 1.0), onset=(0.4, 1.0)),
    # Moderate energy, high pitch, moderate ZCR, high onset (fast speech)
    "happy":   dict(rms=(0.35, 0.75), pitch=(0.45, 0.85), zcr=(0.35, 0.75),
                    centroid=(0.4, 0.8), mfcc_var=(0.35, 0.75), onset=(0.5, 1.0)),
    # Low energy, low pitch, low ZCR, low spectral centroid (slow, monotone)
    "sad":     dict(rms=(0.0, 0.4), pitch=(0.0, 0.45), zcr=(0.0, 0.4),
                    centroid=(0.0, 0.45), mfcc_var=(0.0, 0.4), onset=(0.0, 0.4)),
    # Low-moderate energy, high pitch variation, high ZCR (breathy, trembling)
    "fearful": dict(rms=(0.1, 0.5), pitch=(0.5, 1.0), zcr=(0.45, 0.85),
                    centroid=(0.4, 0.8), mfcc_var=(0.55, 1.0), onset=(0.3, 0.7)),
    # Sudden burst of energy then drop, high onset, moderate pitch
    "surprise": dict(rms=(0.3, 0.8), pitch=(0.5, 1.0), zcr=(0.4, 0.8),
                     centroid=(0.45, 0.85), mfcc_var=(0.45, 0.85), onset=(0.6, 1.0)),
    # Low energy, low-mid pitch, low ZCR, low onset (monotone, steady)
    "neutral":  dict(rms=(0.15, 0.55), pitch=(0.2, 0.6), zcr=(0.1, 0.5),
                     centroid=(0.2, 0.6), mfcc_var=(0.1, 0.5), onset=(0.2, 0.6)),
}

FEATURE_NAMES = ["rms", "pitch", "zcr", "centroid", "mfcc_var", "onset"]


def _extract_features(audio_path: str) -> dict | None:
    """
    Extracts 6 acoustic features from an audio file.
    Returns a dict of raw (un-normalized) feature values, or None on error.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)

        if len(y) == 0:
            return None

        # 1. RMS Energy (Loudness)
        rms = float(np.mean(librosa.feature.rms(y=y)))

        # 2. Fundamental Frequency / Pitch (via piptrack)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        voiced = pitches[magnitudes > np.percentile(magnitudes, 75)]
        pitch = float(np.mean(voiced)) if len(voiced) > 0 else 0.0

        # 3. Zero Crossing Rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

        # 4. Spectral Centroid (brightness)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # 5. MFCC Variance (emotional expressiveness / speech variability)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

        # 6. Onset Strength (speaking rate proxy)
        onset = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))

        return {
            "rms": rms,
            "pitch": pitch,
            "zcr": zcr,
            "centroid": centroid,
            "mfcc_var": mfcc_var,
            "onset": onset
        }

    except Exception as e:
        print(f"[Voice] Feature extraction error: {e}")
        return None


def _normalize_features(raw: dict) -> dict:
    """
    Normalize raw feature values to [0, 1] using known realistic ranges.
    These ranges are based on empirical speech data across typical recordings.
    """
    ranges = {
        "rms":      (0.0,    0.25),
        "pitch":    (0.0,    4000.0),
        "zcr":      (0.0,    0.3),
        "centroid": (0.0,    8000.0),
        "mfcc_var": (0.0,    2000.0),
        "onset":    (0.0,    10.0),
    }
    normalized = {}
    for feat, val in raw.items():
        lo, hi = ranges[feat]
        norm = (val - lo) / (hi - lo) if (hi - lo) > 0 else 0.0
        normalized[feat] = float(np.clip(norm, 0.0, 1.0))
    return normalized


def _score_profile(norm_features: dict, profile: dict) -> float:
    """
    Computes how well a normalized feature dict matches an emotion profile.
    Score = average of per-feature fit values in [0, 1].
    A feature fits perfectly (1.0) if it's inside the profile range,
    and decreases linearly as it moves outside.
    """
    scores = []
    for feat in FEATURE_NAMES:
        val = norm_features.get(feat, 0.0)
        lo, hi = profile[feat]
        mid = (lo + hi) / 2
        width = (hi - lo) / 2 if (hi - lo) > 0 else 0.01
        if lo <= val <= hi:
            fit = 1.0
        else:
            distance = min(abs(val - lo), abs(val - hi))
            fit = max(0.0, 1.0 - (distance / width))
        scores.append(fit)
    return float(np.mean(scores))


def predict_voice_emotion(audio_file: str) -> str:
    """
    Predicts the dominant emotion from an audio file.

    Strategy:
    1. Extract 6 acoustic features.
    2. Normalize all features to [0, 1] using known speech ranges.
    3. Score against all emotion profiles using a range-fit metric.
    4. Return the highest-scoring emotion.
    """
    if not os.path.exists(audio_file):
        return f"Error: File '{audio_file}' not found."

    raw_features = _extract_features(audio_file)
    if raw_features is None:
        return "Error: Could not extract audio features."

    norm_features = _normalize_features(raw_features)

    # Score all emotions
    emotion_scores = {
        emotion: _score_profile(norm_features, profile)
        for emotion, profile in EMOTION_PROFILES.items()
    }

    best_emotion = max(emotion_scores, key=emotion_scores.get)
    return best_emotion


def predict_voice_emotion_with_scores(audio_file: str) -> tuple[str, dict]:
    """
    Like predict_voice_emotion but also returns a dict of confidence scores
    for each emotion (useful for dashboard display and weighted fusion).
    """
    if not os.path.exists(audio_file):
        return f"Error: File not found.", {}

    raw_features = _extract_features(audio_file)
    if raw_features is None:
        return "Error: Could not extract audio features.", {}

    norm_features = _normalize_features(raw_features)

    emotion_scores = {
        emotion: _score_profile(norm_features, profile)
        for emotion, profile in EMOTION_PROFILES.items()
    }

    best_emotion = max(emotion_scores, key=emotion_scores.get)

    # Convert to percentage-style confidence
    total = sum(emotion_scores.values())
    confidence = {e: round((s / total) * 100, 1) if total > 0 else 0.0
                  for e, s in emotion_scores.items()}

    return best_emotion, confidence


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "temp_audio.wav"
    emotion, scores = predict_voice_emotion_with_scores(path)
    print(f"\nPredicted Voice Emotion: {emotion.upper()}")
    print("\nConfidence Breakdown:")
    for e, s in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "=" * int(s / 5)
        print(f"  {e:<10} {s:5.1f}%  [{bar}]")
