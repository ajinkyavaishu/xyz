import sys
sys.path.append('.')
from text_detection.predict_emotion import predict_text_emotion

print("Testing Text:")
print(predict_text_emotion("I am so angry right now"))

from voice_detection.predict_emotion import predict_voice_emotion
print("Testing Voice with non-existent file:")
print(predict_voice_emotion("non_existent.wav"))
