import librosa
import numpy as np
import scipy.fft as fft
from sklearn import preprocessing
import matplotlib.pyplot as plt
import io
import base64

class AudioPreprocessing:
    def __init__(self, audio_path,  frame_size = 2048, sameple_rate = 22050, n_fft = 2048, n_mels = 128, frame_rate = 31.25, ) -> None:
        self.frame_size = frame_size
        self.hop_length = int(sameple_rate // frame_rate)
        self.sameple_rate = 22050
        self.n_fft = 2048
        self.n_mels = 149
        self.audio, self.sr = self.__random_10s(audio_path)

    def __random_10s(self, audio_path):
      signal, sr = librosa.load(audio_path)
      section_start = np.random.randint(0, signal.shape[0] - 10 * 22050)
      section_end = section_start + 10 * 22050
      section = signal[section_start:section_end]
      return section, sr
    
    def extract_mel(self):
        spectrogram = librosa.feature.melspectrogram(y=self.audio, sr=self.sameple_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        spectrogram_db = np.transpose(spectrogram_db)

        return spectrogram_db

    def extract_waveform(self):
        return self.audio
    
    def extract_spectral_centroid(self, normalize = False):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)
        if normalize:
          return sklearn.preprocessing.minmax_scale(y=self.audio, axis=0)
        return spectral_centroid

    def extract_mfcc(self):
        return librosa.feature.mfcc(y=self.audio, sr=self.sr)
    
    def extract_chromagram(self):
        return librosa.feature.chroma_stft(y=self.audio, sr=self.sr)
    
    def extract_peak_value(self):
        return librosa.amplitude_to_db(np.abs(self.audio))
    
    def extract_rms_value(self):
        return librosa.feature.rms(y=self.audio)
    
    def extract_bpm_value(self, is_confidence = None):
        return librosa.beat.beat_track(y=self.audio, sr=self.sr, units = 'confidence') if is_confidence else librosa.beat.beat_track(y=self.audio, sr=self.sr)

    def extract_pitch(self):
        return librosa.piptrack(y=self.audio, sr=self.sr)

    def extract_waveform_base64(self):
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y=self.audio, sr=self.sr)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        wave_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()
        return wave_base64
    
    def extract_fft_base64(self):
        X = np.fft.fft(self.audio)
        X_mag = np.absolute(X)
        f = np.linspace(0, self.sr, len(X_mag)) # frequency variable

        plt.figure(figsize=(10, 4))
        plt.plot(f, X_mag)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        wave_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()
        return wave_base64

    def extract_stft_base64(self):
        X = librosa.stft(y=self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S = librosa.amplitude_to_db(abs(X))

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        wave_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()
        return wave_base64

    def extract_cqt_base64(self):
        fmin = librosa.midi_to_hz(36)
        C = librosa.cqt(y=self.audio, sr=self.sr, fmin=fmin, n_bins=72)
        logC = librosa.amplitude_to_db(abs(C))


        plt.figure(figsize=(10, 4))
        librosa.display.specshow(logC, sr=self.sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        wave_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()
        return wave_base64
    
    def extract_chromagram_base64(self):
        chromagram = librosa.feature.chroma_stft(y=self.audio, sr=self.sr, hop_length=self.hop_length)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=self.hop_length, cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        wave_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()
        return wave_base64
    
    def extract_spectral_centroids_base64(self):
        spectral_centroids = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)

        librosa.display.waveshow(y=self.audio, sr=self.sr, alpha=0.4)
        plt.plot(t, preprocessing.minmax_scale(spectral_centroids, axis=0), color='r')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        wave_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()
        return wave_base64

    def extract_mel_base64(self):
        S = librosa.feature.melspectrogram(y= self.audio, sr=self.sr, n_fft=4096, hop_length=self.hop_length)
        logS = librosa.power_to_db(abs(S))
        
        # Plot the mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(logS, x_axis='time', y_axis='mel', sr=self.sr, hop_length=self.hop_length)
        plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel Spectrogram')
        plt.tight_layout()

        # Convert the plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        mel_spectrogram_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to avoid displaying it
        plt.close()

        return mel_spectrogram_base64
    # def estimate_key(self):

    #     # Extract the chromagram
    #     chromagram = librosa.feature.chroma_cqt(y=self.audio, sr=self.sr)

    #     # Compute the mean value along the time axis
    #     chroma_mean = chromagram.mean(axis=1)

    #     # Find the key/mode index
    #     # key_index = chroma_mean.argmax()

    #     # Get the corresponding key/mode label
    #     key_midi = librosa.core.note_to_midi(chroma_mean)
    #     key_label = librosa.core.midi_to_note(key_midi)

    #     # Return the estimated key/mode
    #     return key_label