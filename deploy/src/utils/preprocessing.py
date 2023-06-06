import librosa
import numpy as np

class AudioPreprocessing:
    def __init__(self, frame_size = 2048, sameple_rate = 22050, n_fft = 2048, n_mels = 128, frame_rate = 31.25) -> None:
        self.frame_size = frame_size
        self.hop_length = int(sameple_rate // frame_rate)
        self.sameple_rate = 22050
        self.n_fft = 2048
        self.n_mels = 149


    def __random_10s(self, signal):
      section_start = np.random.randint(0, signal.shape[0] - 10 * 22050)
      section_end = section_start + 10 * 22050
      section = signal[section_start:section_end]
      return section
    
    def extract_mel(self, audio_path):
        audio, sr = librosa.load(audio_path)
        signal = self.__random_10s(audio)
        spectrogram = librosa.feature.melspectrogram(y=signal, sr=self.sameple_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        spectrogram_db = np.transpose(spectrogram_db)

        return spectrogram_db
       