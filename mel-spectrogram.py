import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt


signal, sr = librosa.load('.\\audio-files\\Behind Her Eyes (widerberg Remix).mp3', sr=None)
fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

# =========================
# 1) WAVEFORM (time-domain)
# =========================
librosa.display.waveshow(signal, sr=sr, ax=axs[0])
# Set x-axis limits in seconds:
# from 0 to total duration of the signal (len(signal) / sr)
axs[0].set_xlim(left=0.0, right=len(signal)*(1/sr)) 

axs[0].set_title('Waveform', fontsize=14)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Amplitude')
# Hide x-axis labels where they are redundant
axs[0].label_outer()


# =========================
# 2) MEL-SPECTROGRAM (in dB)
# =========================

# n_fft: number of samples per FFT window
n_fft = 2048

# hop_length: number of samples to shift between successive FFT windows
hop_length = 512

# Compute the Mel-spectrogram
mel_signal = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft
)

# Magnitude
spectrogram = np.abs(mel_signal)
# Convert power to decibels (log scale), referenced to the maximum
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

# Display the Mel-spectrogram:
img = librosa.display.specshow(
    power_to_db,
    sr=sr,
    x_axis='time',
    y_axis='mel',
    cmap='magma',
    hop_length=hop_length,
    ax=axs[1]
)

axs[1].set_title('Mel-Spectrogram', fontsize=14)
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Mel frequency')
axs[1].label_outer()

# Uncomment to schow the colorbar (for the dB scale)
#cbar = fig.colorbar(img, ax=axs[1])
#cbar.set_label('Power [dB]')

# =========================
# 3) SPECTRAL CENTROID
# =========================
centroid = librosa.feature.spectral_centroid(
    y=signal,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length
)

times_centroid = librosa.times_like(centroid, sr=sr, hop_length=hop_length)

axs[2].plot(times_centroid, centroid[0], label='Spectral Centroid (Hz)')
axs[2].set_ylabel('Centroid (Hz)')
axs[2].legend()

# Show the final figure
plt.show()