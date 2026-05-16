from deepface import DeepFace
import os

print("Downloading DeepFace Emotion model weights...")
try:
    DeepFace.build_model('Emotion')
    print("Successfully downloaded Emotion model.")
except Exception as e:
    print(f"Error downloading model: {e}")
