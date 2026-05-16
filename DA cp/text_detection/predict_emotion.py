import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download lexicon on first run silently
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ─────────────────────────────────────────────
# Emotion Keyword Lexicon
# Each emotion has a list of strong signal words.
# ─────────────────────────────────────────────
EMOTION_KEYWORDS = {
    "happy": [
        "happy", "joy", "joyful", "excited", "amazing", "fantastic", "wonderful",
        "great", "love", "awesome", "delighted", "thrilled", "cheerful", "elated",
        "ecstatic", "bliss", "blessed", "grateful", "smile", "laugh", "fun",
        "beautiful", "brilliant", "fantastic", "glad", "pleased", "enjoy",
        "celebration", "celebrate", "yay", "woohoo", "best day", "best",
        "incredible", "magnificent", "superb", "excellent", "perfect", "hooray"
    ],
    "sad": [
        "sad", "unhappy", "depressed", "depression", "grief", "sorrow", "cry",
        "crying", "tears", "heartbroken", "heartbreak", "miserable", "lonely",
        "alone", "hopeless", "devastated", "disappointed", "upset", "gloomy",
        "melancholy", "mourn", "mourning", "regret", "sorry", "pain", "hurt",
        "broken", "lost", "miss", "missing", "wished", "wish things were different",
        "empty", "nothing", "nobody cares", "helpless", "worthless", "tired of"
    ],
    "angry": [
        "angry", "anger", "furious", "rage", "mad", "hate", "hatred", "outraged",
        "irritated", "annoyed", "frustrated", "infuriated", "livid", "hostile",
        "aggressive", "resentful", "bitter", "disgusted", "fed up", "sick of",
        "can't stand", "ridiculous", "stupid", "idiot", "damn", "hell", "worst",
        "terrible", "awful", "disgusting", "pathetic", "unacceptable", "unfair",
        "nonsense", "bull", "rubbish", "screw this", "not okay"
    ],
    "fearful": [
        "scared", "afraid", "frightened", "terrified", "terror", "fear", "fearful",
        "anxious", "anxiety", "nervous", "panic", "panicking", "dread", "dreading",
        "horrified", "horror", "uneasy", "worried", "worrying", "trembling",
        "shaking", "phobia", "nightmare", "threatening", "danger", "unsafe",
        "overwhelming", "overwhelmed", "helpless", "vulnerable", "paralyzed"
    ],
    "surprise": [
        "surprised", "surprise", "shocked", "shocking", "astonished", "astonishing",
        "amazed", "amazing", "unexpected", "unbelievable", "incredible", "wow",
        "omg", "oh my god", "what the", "no way", "seriously", "really",
        "i can't believe", "never expected", "didn't see that coming", "blown away",
        "jaw dropped", "speechless", "stunning", "startled", "whoa"
    ],
    "disgust": [
        "disgusting", "disgust", "gross", "revolting", "repulsive", "nauseating",
        "sick", "nasty", "vile", "filthy", "yuck", "eww", "ew", "horrible",
        "appalling", "repelled", "loathe", "loathing", "detest", "abhor",
        "stomach-turning", "makes me sick", "can't stomach"
    ],
    "neutral": [
        "okay", "ok", "fine", "alright", "sure", "whatever", "maybe", "perhaps",
        "normal", "ordinary", "average", "usual", "typical", "nothing special",
        "so-so", "not bad", "not great"
    ]
}

# Negation words that flip the detected emotion
NEGATIONS = {"not", "no", "never", "neither", "nor", "don't", "doesn't",
             "didn't", "isn't", "wasn't", "can't", "couldn't", "won't",
             "wouldn't", "shouldn't", "haven't", "hadn't", "barely", "hardly"}

# Emotions that flip to their opposites when negated
NEGATION_FLIP = {
    "happy": "sad",
    "sad": "happy",
    "angry": "neutral",
    "fearful": "neutral",
    "disgust": "neutral",
    "surprise": "neutral",
    "neutral": "neutral"
}


def _tokenize(text: str) -> list:
    """Simple word tokenizer."""
    return re.findall(r"[a-z']+", text.lower())


def _detect_intensity_boosters(text: str) -> float:
    """
    Returns a multiplier based on intensity signals:
    - Exclamation marks
    - ALL CAPS words
    - Repeated punctuation (!!!, ???)
    """
    booster = 1.0
    # Exclamation marks
    excl_count = text.count("!")
    booster += min(excl_count * 0.15, 0.45)
    # ALL CAPS words (not single letters)
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    booster += min(len(caps_words) * 0.1, 0.3)
    # Repeated punctuation
    if re.search(r'[!?]{2,}', text):
        booster += 0.2
    return booster


def _score_emotions(tokens: list, booster: float) -> dict:
    """
    Scores every emotion based on keyword matches in token list.
    Returns a dict of {emotion: score}.
    """
    scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS}
    negation_active = False

    for i, token in enumerate(tokens):
        # Check for negation window (negation affects next 3 words)
        if token in NEGATIONS:
            negation_active = True
            negation_window = 3
            continue

        if negation_active:
            negation_window -= 1
            if negation_window <= 0:
                negation_active = False

        for emotion, keywords in EMOTION_KEYWORDS.items():
            if token in keywords:
                match_score = booster
                if negation_active:
                    # Flip the emotion on negation
                    flipped = NEGATION_FLIP.get(emotion, emotion)
                    scores[flipped] += match_score
                else:
                    scores[emotion] += match_score

    return scores


def predict_text_emotion(text: str) -> str:
    """
    Predicts the dominant emotion from a text string.

    Strategy:
    1. Run keyword-based scoring across all 7 emotion categories.
    2. Use VADER sentiment as a tiebreaker / confidence booster.
    3. Return the highest-scoring emotion.
    """
    if not text or not text.strip():
        return "neutral"

    try:
        clean_text = text.strip()
        tokens = _tokenize(clean_text)
        booster = _detect_intensity_boosters(clean_text)

        # ── Step 1: Keyword scoring ──────────────────────────────
        scores = _score_emotions(tokens, booster)

        # ── Step 2: VADER as a booster ───────────────────────────
        sia = SentimentIntensityAnalyzer()
        vader = sia.polarity_scores(clean_text)
        compound = vader["compound"]

        # Boost happy/sad/angry based on VADER compound
        if compound >= 0.3:
            scores["happy"] += compound * 0.5
        elif compound <= -0.3:
            # Differentiate sad vs angry using neg score
            scores["sad"] += abs(compound) * 0.3
            scores["angry"] += vader["neg"] * 0.3

        # ── Step 3: Repeated-letter heuristic (excitement) ──────
        # e.g. "yesss", "hiiii", "omgggg"
        if re.search(r'([a-z])\1{2,}', clean_text.lower()):
            scores["happy"] += 0.5

        # ── Step 4: Short, blunt text heuristic ─────────────────
        word_count = len(tokens)
        if word_count <= 3 and compound <= 0.0:
            scores["sad"] += 0.3
            scores["neutral"] += 0.2

        # ── Step 5: Pick winner ──────────────────────────────────
        # If no keywords matched at all, fall back to VADER mapping
        total_keyword_score = sum(scores.values())
        if total_keyword_score == 0:
            if compound >= 0.5:
                return "happy"
            elif compound <= -0.5:
                return "sad"
            else:
                return "neutral"

        best_emotion = max(scores, key=scores.get)
        return best_emotion

    except Exception as e:
        return f"Error predicting emotion: {str(e)}"


if __name__ == "__main__":
    test_cases = [
        "I am so happy today!",
        "I feel extremely sad and alone",
        "I am furious and can't stand this anymore!",
        "This is shocking, I didn't expect that at all!",
        "That is absolutely disgusting",
        "I'm terrified and can't stop shaking",
        "It's just a normal day, nothing special",
        "I'm not happy at all",
        "YESSSS THIS IS AMAZING!!!",
        "ok fine whatever",
    ]
    print("=" * 50)
    for t in test_cases:
        print(f"  Text    : {t}")
        print(f"  Emotion : {predict_text_emotion(t)}")
        print("-" * 50)
