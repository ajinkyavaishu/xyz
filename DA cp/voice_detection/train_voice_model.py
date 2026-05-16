# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
import numpy as np
import os
import pickle
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score

# RAVDESS Dataset Emotion Mapping
# RAVDESS names files like: 03-01-01-01-01-01-01.wav
# The 3rd identifier represents the emotion:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# We will focus on these for our system
OBSERVED_EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'fearful']

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """
    Extracts MFCC, Chroma, and Mel features from an audio file.
    """
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
            
        return result
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

def load_data(data_path, test_size=0.2):
    """
    Loads RAVDESS dataset audio files and extracts features.
    Expects data_path to contain Actor_* folders.
    """
    x, y = [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith('.wav'):
                continue
                
            file_path = os.path.join(root, file)
            # Extract emotion from filename
            emotion_code = file.split("-")[2]
            emotion = EMOTIONS.get(emotion_code)
            
            if emotion not in OBSERVED_EMOTIONS:
                continue
                
            feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
            if feature is not None:
                x.append(feature)
                y.append(emotion)
                
    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)

def train_and_save_voice_model():
    """
    Trains a Random Forest classifier for voice emotion detection.
    """
    data_path = '../datasets/ravdess/' # Ensure dataset is placed here
    
    if not os.path.exists(data_path):
        print(f"Dataset path {data_path} not found. Please download RAVDESS and place it there.")
        print("Training with dummy data for demonstration purposes...")
        
        # Dummy data so the script runs and generates a model for testing
        X_train = np.random.rand(100, 180) # 180 features from MFCC+Chroma+Mel
        y_train = np.random.choice(OBSERVED_EMOTIONS, 100)
        X_test = np.random.rand(20, 180)
        y_test = np.random.choice(OBSERVED_EMOTIONS, 20)
    else:
        print("Loading audio features... This may take a while.")
        X_train, X_test, y_train, y_test = load_data(data_path, test_size=0.25)
        
    print(f"Features extracted: {X_train.shape[1]}")
    
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Save Model
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'voice_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Voice emotion model saved to {model_path}!")

if __name__ == "__main__":
    train_and_save_voice_model()
