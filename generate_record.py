import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample
import matplotlib.pyplot as plt


cutoff = 1000000

sr, data = wavfile.read("test.wav")  # data ist ein NumPy-Array

if data.ndim > 1:
    data = data[:, 0]

samplerate = 44100  # z.B. 8 kHz
time_cutoff = samplerate * 10

# Anzahl der Samples neu berechnen
num_samples = int(len(data) * samplerate / samplerate)

# Resample
resampled_data = resample(data, num_samples)

def lowpass_filter_butter(data, cutoff, fs, order=100):
    sos = butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
    return sosfilt(sos, data)

#filtered_data = lowpass_filter_butter(data, cutoff, samplerate)
waveform_array = np.array(data)[0:time_cutoff]
waveform_array_normalized = (waveform_array / max(waveform_array) + 1)/2
record_matrix = waveform_array_normalized.reshape(10, -1)
record_matrix[1::2] = record_matrix[1::2, ::-1]
print(record_matrix.shape)


plt.plot(waveform_array_normalized)
plt.show()



wavfile.write("filtered_audio.wav", samplerate, waveform_array.astype(np.int16))