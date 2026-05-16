import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def train_and_save_text_model():
    """
    Trains a simple text emotion detection model using a Naive Bayes classifier
    and saves it to the models/ folder.
    """
    # Sample Dataset (for demonstration purposes)
    # In a real scenario, you'd load a CSV like Twitter Emotion Dataset
    data = {
        'text': [
            "I feel amazing today", "This is the best day ever", "I am so happy",
            "I am feeling very sad", "This is so depressing", "I want to cry",
            "I am extremely angry!", "This is infuriating", "I hate this",
            "Wow, I didn't expect that!", "What a surprise!", "That is shocking",
            "I am so scared", "This is terrifying", "I feel fearful",
            "It's just an ordinary day", "Nothing special is happening", "I'm okay"
        ],
        'emotion': [
            "happy", "happy", "happy",
            "sad", "sad", "sad",
            "angry", "angry", "angry",
            "surprise", "surprise", "surprise",
            "fear", "fear", "fear",
            "neutral", "neutral", "neutral"
        ]
    }
    
    df = pd.DataFrame(data)
    X = df['text']
    y = df['emotion']
    
    # Simple Pipeline: Vectorize -> TF-IDF -> Naive Bayes Classifier
    text_clf = Pipeline([
        ('vect', CountVectorizer(lowercase=True, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    
    # Train
    text_clf.fit(X, y)
    
    # Save Model
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'text_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(text_clf, f)
        
    print(f"Text emotion model trained successfully and saved to {model_path}!")

if __name__ == "__main__":
    train_and_save_text_model()
