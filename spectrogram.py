import librosa
import numpy as np

# I often start by loading a short audio clip to test my processing chain
audio, sample_rate = librosa.load('filtered_audio.wav', sr=44100)
duration = len(audio) / sample_rate
print(f"Loaded {duration:.2f} seconds of audio at {sample_rate} Hz sample rate")

# Sometimes I need to handle stereo files by converting to mono
if len(audio.shape) > 1:
    audio = librosa.to_mono(audio)
    print("Converted stereo to mono for consistent processing")

import matplotlib.pyplot as plt

# Computing the spectrogram reveals the frequency landscape
spectrogram = librosa.stft(audio)
magnitude_spectrum = np.abs(spectrogram)
decibel_spectrogram = librosa.amplitude_to_db(magnitude_spectrum)

# I often customize the visualization for better clarity
plt.figure(figsize=(12, 6))
librosa.display.specshow(decibel_spectrogram, sr=sample_rate,
                         x_axis='time', y_axis='log',
                         hop_length=512)
plt.colorbar(label='Decibels (dB)')
plt.title('Frequency Content Over Time')
plt.tight_layout()
plt.show()

# For specific frequency analysis, I sometimes focus on particular ranges
frequencies = librosa.fft_frequencies(sr=sample_rate)
mid_range = (frequencies > 200) & (frequencies < 2000)
print(f"Mid-range frequencies contain {np.mean(magnitude_spectrum[mid_range, :]):.4f} average magnitude")