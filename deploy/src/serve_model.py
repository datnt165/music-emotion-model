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

    audio_processor = AudioPreprocessing(audio_path=audio_file)
    feature = audio_processor.extract_mel()
    feature = [np.expand_dims(feature, axis=-1)]
    middle_feature, input = model.predict(feature)
    emotion = emotion_model.predict(input)
    mel_spec = audio_processor.extract_mel()
    spectral_centroid = audio_processor.extract_spectral_centroid()
    mfcc = audio_processor.extract_mfcc()
    chromagram = audio_processor.extract_chromagram()
    peak_value = audio_processor.extract_peak_value()
    rms_value = audio_processor.extract_rms_value() 
    mel_waveform = audio_processor.extract_mel_base64()
    audio_waveform = audio_processor.extract_waveform_base64()
    fft_waveform = audio_processor.extract_fft_base64()
    stft_waveform = audio_processor.extract_stft_base64()
    cqt_waveform = audio_processor.extract_cqt_base64()
    chromagram_waveform = audio_processor.extract_chromagram_base64()
    spectral_centroid_waveform = audio_processor.extract_spectral_centroids_base64()
    # bpm_value = audio_processor.extract_bpm_value()
    # pitch = audio_processor.extract_pitch()
    # estimate_key = audio_processor.estimate_key()

    return {
        'song': name, 
        'middle_level': middle_feature, 
        'emotion': emotion, 
        'mel_waveform': mel_waveform,
        'audio_waveform': audio_waveform,
        'fft_waveform': fft_waveform,
        'stft_waveform': stft_waveform,
        'cqt_waveform': cqt_waveform,
        'chromagram_waveform': chromagram_waveform,
        'spectral_centroid_waveform': spectral_centroid_waveform,
        # 'mel_spec': mel_spec.tolist(), 
        # 'spectral_centroid': spectral_centroid.tolist(),
        # 'mfcc' : mfcc.tolist(),
        # 'chromagram': chromagram.tolist(),
        # 'peak_value': peak_value.tolist(),
        # 'rms_value': rms_value.tolist(),
        # 'bpm_value': bpm_value[0],
        # 'pitch': {
        #     'tempo': pitch[0], 
        #     'beat_frames': pitch[1], 
        # },
    }

