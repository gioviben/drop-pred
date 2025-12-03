import numpy as np
import librosa
import matplotlib.pyplot as plt
import math

signal, sr = librosa.load('.\\audio-files\\Behind Her Eyes (widerberg Remix).mp3', sr=None)

n_mels = 128
n_fft = 2048
hop_length = 512
mel_signal = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft,
    n_mels = n_mels
)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)


centroid = librosa.feature.spectral_centroid(
    y=signal,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length
)

# Normalizzazione asse Y (Centroid) su [0, 1]
centroid_norm = centroid / (sr / 2)

# Calcolo parametri temporali
delta_frame = hop_length / sr
total_frames = centroid_norm.shape[1]

# --- 3. PARAMETRI FINESTRA ---
duration_curr_win = 3.0       # Durata finestra analisi (secondi)
time_between_wins = 0.05      # Step avanzamento (secondi)

# Convertiamo durate in numero di frames
n_frames_curr_win = int(round(duration_curr_win / delta_frame))
step_frames = int(round(time_between_wins / delta_frame))

start_idx = 0

# --- 4. CICLO DI ANALISI ---
print(f"Analisi start... Frames per finestra: {n_frames_curr_win}")

while start_idx + n_frames_curr_win <= total_frames:
    end_idx = start_idx + n_frames_curr_win
    
    # a. Estrazione Centroid (Y)
    # centroid è (1, frames), usiamo [0, ...] per averlo 1D
    win_centroid = centroid_norm[0, start_idx:end_idx]
    
    # b. Creazione asse Tempo Normalizzato (X) in [0, 1]
    # È più robusto usare linspace che fare calcoli sui timestamp originali
    win_time_norm = np.linspace(0, 1, num=len(win_centroid))
    
    # c. Controllo Sicurezza (es. se c'è silenzio assoluto o NaN)
    if len(win_centroid) == 0 or np.all(win_centroid == 0):
        m_normalized = 0.5 # Consideriamo "piatto" se vuoto
    else:
        # d. Regressione Lineare
        # Fit di grado 1 (retta): y = mx + b
        m, b = np.polyfit(win_time_norm, win_centroid, 1)
        sensitivity = 50.0  # Prova con 10, 20, 50

        m_scaled = m * sensitivity
        # e. Normalizzazione Angolare
        angle_rad = np.arctan(m_scaled)
        m_normalized = (angle_rad + (np.pi / 2)) / np.pi

    # --- DEBUG / VISUALIZZAZIONE ---
    # Stampiamo solo se c'è una pendenza rilevante (es. > 0.6 o < 0.4)
    # 0.5 è piatto.
    if m_normalized >= 0.96: 
        curr_time_sec = start_idx * delta_frame
        print(f"Time: {curr_time_sec:.2f}s | Slope Norm: {m_normalized:.4f} (In salita ripida)")
    #if curr_time_sec >= 120:
        # Plotting
        plt.figure(figsize=(6, 4))
        plt.scatter(win_time_norm, win_centroid, color='blue', alpha=0.5, s=10, label='Dati Centroid')
        plt.plot(win_time_norm, win_time_norm * m + b, color='red', linewidth=2, label='Regressione')
        plt.title(f"Finestra a {curr_time_sec:.2f}s - Slope: {m_normalized:.2f}")
        plt.ylim(0, 1) # Asse Y sempre fisso tra 0 e 1 (Nyquist)
        plt.xlim(0, 1) # Asse X sempre fisso tra 0 e 1 (Finestra temporale)
        plt.legend()
        plt.show() 
        # Nota: plt.show() blocca. Se vuoi che scorra da solo usa:
        #plt.pause(0.08)
        #plt.close()
        #plt.clf()
        
    # Avanzamento
    start_idx += step_frames