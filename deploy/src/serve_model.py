from src.middle_model.model import MiddlePredictor
from src.emotion_model.model import EmotionPredictor
from src.utils.preprocessing import AudioPreprocessing
from src.config import middle_prediction, emotion_prediction
import numpy as np
from io import BytesIO

model = None
emotion_model = None

def load_model():
    print("Model loading.....")
    model = MiddlePredictor(path_to_labels=middle_prediction['path_to_labels'], path_to_model=middle_prediction['path_to_model'])
    emotion_model = EmotionPredictor(path_to_labels=emotion_prediction['path_to_labels'], path_to_model=emotion_prediction['path_to_model'])
    print("!!! Completed")

    return model, emotion_model

def read_audio_file(file):
    return BytesIO(file)

def predict(audio_file, name):
    global model, emotion_model
    if model is None:
        model, emotion_model = load_model()

    audio_processor = AudioPreprocessing()
    feature = audio_processor.extract_mel(audio_file)
    feature = [np.expand_dims(feature, axis=-1)]
    result1, input = model.predict(feature)
    result2 = emotion_model.predict(input)
    

    return {'song': name, 'middle_level': result1, 'emotion': result2}

