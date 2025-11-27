import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import math

def to_sec(min):        
    sec = (min * 100) % 100
    return int(min) * 60 + int(sec)

signal, sr = librosa.load('.\\audio-files\\Behind Her Eyes (widerberg Remix).mp3', sr=None)
fig, axs = plt.subplots(2, 1, figsize=(20, 5))

n_mels = 128

# n_fft: number of samples per FFT window
n_fft = 2048

# hop_length: number of samples to shift between successive FFT windows
hop_length = 512

# Compute the Mel-spectrogram
mel_signal = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft,
    n_mels = n_mels
)

# Magnitude
spectrogram = np.abs(mel_signal)
# Convert power to decibels (log scale), referenced to the maximum
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

delta_frame = hop_length / sr

end_column_at_t = math.floor((to_sec(5.43) * sr) / hop_length)  #4.23

finestra_sec = 3.0
num_frames = finestra_sec / delta_frame 

start_column = round(end_column_at_t - num_frames)


#print(power_to_db.shape)
power_to_db = power_to_db[:, start_column:end_column_at_t]     
#print(power_to_db.shape)
#print(power_to_db.size)


mel_freqs = librosa.mel_frequencies(n_mels=n_mels)

cutoff_f = 200
mask = mel_freqs <= cutoff_f
print()
print(mask)
print(mel_freqs[mask])

power_to_db_mask = power_to_db[mask, :]
print()
print(power_to_db_mask.shape)

colonna = round(0.5 / delta_frame)
print(power_to_db_mask[:, colonna])

print(f"Al secondo: {colonna * delta_frame}")

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

# Show the final figure
plt.show()






'''
num_samples = len(signal)
duration = num_samples / sr

print("Numero di samples:", num_samples)
print("Durata in secondi:", duration)

win_size = 128

win_number = math.ceil(num_samples/win_size)

print("Numero di finestre:", win_number)

for i in range(win_number):
    start_idx = i*win_size
    end_idx = start_idx + win_size
    y = signal[start_idx:end_idx]
    fig, axs = plt.subplots(2, 1, figsize=(20, 5))
    # =========================
    # 1) WAVEFORM (time-domain)
    # =========================
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    # Set x-axis limits in seconds:
    # from 0 to total duration of the signal (len(signal) / sr)
    axs[0].set_xlim(left=0.0, right=len(y)*(1/sr)) 

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
        y=y,
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

    # Show the final figure
    plt.show()
    input("pause")
'''