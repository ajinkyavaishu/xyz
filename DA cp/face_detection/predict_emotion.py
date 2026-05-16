import cv2
import numpy as np
from collections import deque
from deepface import DeepFace

# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoothing buffer — stores last N emotion score dicts.
# This smooths out flicker when using the camera in real-time mode.
# ─────────────────────────────────────────────────────────────────────────────
_SMOOTHING_WINDOW = 5
_emotion_history: deque = deque(maxlen=_SMOOTHING_WINDOW)

# ─────────────────────────────────────────────────────────────────────────────
# Label normalization — DeepFace returns its own labels. We standardize them
# to match the vocabulary used by Voice and Text modules:
#   happy | sad | angry | neutral | fearful | surprise | disgust
# ─────────────────────────────────────────────────────────────────────────────
DEEPFACE_TO_STANDARD = {
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "angry",
    "neutral":   "neutral",
    "fear":      "fearful",
    "disgust":   "disgust",
    "surprise":  "surprise",
    "surprised": "surprise",
    "fearful":   "fearful",
}

# Canonical emotion set used across all modules
CANONICAL_EMOTIONS = {"happy", "sad", "angry", "neutral", "fearful", "surprise", "disgust"}


def _normalize_label(label: str) -> str:
    """Converts a DeepFace emotion label to the project's standard vocabulary."""
    return DEEPFACE_TO_STANDARD.get(label.lower(), "neutral")


def _get_smoothed_emotion(current_scores: dict) -> str:
    """
    Adds current frame's scores to history buffer and returns the
    smoothed dominant emotion (average scores across last N frames).
    """
    _emotion_history.append(current_scores)

    # Average scores across history
    avg_scores = {e: 0.0 for e in CANONICAL_EMOTIONS}
    for frame_scores in _emotion_history:
        for emotion, score in frame_scores.items():
            std_label = _normalize_label(emotion)
            if std_label in avg_scores:
                avg_scores[std_label] += score

    # Normalize averages
    n = len(_emotion_history)
    avg_scores = {e: v / n for e, v in avg_scores.items()}

    return max(avg_scores, key=avg_scores.get)


def detect_face_emotion(frame: np.ndarray, use_smoothing: bool = True):
    """
    Detects faces in a frame and predicts emotion using DeepFace.
    Uses full confidence score vector + temporal smoothing to reduce noise.

    Args:
        frame: BGR or RGB numpy image array
        use_smoothing: If True, averages predictions over last 5 frames

    Returns:
        (annotated_frame, dominant_emotion, confidence_dict)
    """
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        # DeepFace can return a list (multiple faces) or a single dict
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # Raw emotion confidence scores from DeepFace (sum to ~100)
        raw_scores = result.get('emotion', {})

        # Dominant emotion (optionally smoothed)
        if use_smoothing:
            dominant_emotion = _get_smoothed_emotion(raw_scores)
        else:
            dominant_emotion = _normalize_label(result.get('dominant_emotion', 'neutral'))

        # Normalize scores to standard vocabulary
        std_scores = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for label, score in raw_scores.items():
            std_label = _normalize_label(label)
            if std_label in std_scores:
                std_scores[std_label] += score

        # Draw bounding box and label on frame
        region = result.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)

        if w > 0 and h > 0:
            confidence = std_scores.get(dominant_emotion, 0.0)

            # Color reflects emotion (green=happy, blue=sad, red=angry, etc.)
            color_map = {
                "happy":   (0, 255, 100),
                "sad":     (255, 100, 0),
                "angry":   (0, 0, 255),
                "neutral": (200, 200, 200),
                "fearful": (255, 165, 0),
                "surprise":(255, 255, 0),
                "disgust": (100, 0, 200),
            }
            color = color_map.get(dominant_emotion, (0, 255, 0))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{dominant_emotion.upper()} ({confidence:.0f}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return frame, dominant_emotion, std_scores

    except Exception as e:
        print(f"[Face] DeepFace error: {e}")
        empty_scores = {e: 0.0 for e in CANONICAL_EMOTIONS}
        empty_scores["neutral"] = 100.0
        return frame, "neutral", empty_scores


def reset_smoothing():
    """Clears the temporal smoothing history. Call when switching inputs."""
    _emotion_history.clear()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, emotion, scores = detect_face_emotion(frame)
        cv2.imshow('Face Emotion Detection', annotated_frame)

        # Print top-2 emotions in terminal
        top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
        print(f"\rEmotion: {emotion:10s}  | " +
              " | ".join(f"{e}: {s:.0f}%" for e, s in top2), end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
