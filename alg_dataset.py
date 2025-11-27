import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import math

def to_sec(min):        
    sec = (min * 100) % 100
    return int(min) * 60 + int(sec)

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




delta_frame = hop_length / sr

duration_curr_win = 3
duration_fut_win = 3
time_between_wins = 0.05

n_frames_curr_win = round(duration_curr_win / delta_frame)
n_frames_len_fut_win = round(duration_fut_win / delta_frame)

delta_between_curr_wins = round(time_between_wins / delta_frame)

start_idx_curr_win = 0
end_idx_curr_win = 0

start_idx_fut_win = 0
end_idx_fut_win = 0

starting_song_frame = 0
offset = math.ceil((duration_curr_win + duration_fut_win) / delta_frame )
last_song_frame = power_to_db.shape[1] - offset


mel_freqs = librosa.mel_frequencies(n_mels=n_mels)
low_freq_max = 150.0
low_band = np.where(mel_freqs <= low_freq_max)[0] 

global_mean_db = np.mean(power_to_db)

while start_idx_curr_win < last_song_frame:

    end_idx_curr_win = start_idx_curr_win + n_frames_curr_win
    curr_win_mel_spectrogram = power_to_db[:, start_idx_curr_win:end_idx_curr_win]

    start_idx_fut_win = end_idx_curr_win
    end_idx_fut_win = start_idx_fut_win + n_frames_len_fut_win
    fut_win_mel_spectrogram = power_to_db[:, start_idx_fut_win:end_idx_fut_win]

    # 5) Energia media attuale (totale + low)
    curr_win_mean_db = np.mean(curr_win_mel_spectrogram, axis=0)                        #mean between all the bands for each frame
    curr_win_low_mean_db = np.mean(curr_win_mel_spectrogram[low_band, :], axis=0)       #mean between low bands for each frame

    curr_win_med_db = np.median(curr_win_mean_db)                                       # livello "tipico" ora
    curr_win_low_med_db = np.median(curr_win_low_mean_db)
    curr_win_min_db = np.percentile(curr_win_mean_db, 10)                               # parti più vuote
    curr_win_max_db = np.max(curr_win_mean_db)                                          # picco nella finestra

    # 6) Energia futura: prendo il massimo (picco)
    fut_win_mean_db = np.mean(fut_win_mel_spectrogram, axis=0)
    fut_win_low_mean_db = np.mean(fut_win_mel_spectrogram[low_band, :], axis=0)
    
    fut_win_med_db  = np.median(fut_win_mean_db)
    fut_win_max_db  = np.max(fut_win_mean_db)    
    fut_win_low_max_db  = np.max(fut_win_low_mean_db)  # picco low

    low_freq_max=150.0,     # max freq per la banda "low" (kick/bass)
    drop_boost_db=8.0,      # quanto il futuro deve essere + forte (in dB) rispetto all'ora
    low_suppression_db=6.0, # quanto le low devono essere + basse del totale nella finestra
    future_min_db=-30.0,    # energia minima del futuro per dire che c'è davvero un drop

    # ---- Condizioni per y=1 (pre-drop) ----
    # (a) ora: poche basse e energia non enorme

    #la mediana dell’energia totale è più alta della mediana delle low di almeno low_suppression_db
    #interpretazione: le basse sono relativamente “vuote” rispetto al resto dello spettro
    cond_low_suppressed = (curr_win_med_db - curr_win_low_med_db) >= low_suppression_db             #pochi bassi?
    
    #la mediana dell’energia attuale è sotto la media globale del brano
    #interpretazione: questa zona è meno energica rispetto al resto della traccia (build-up / breakdown)
    cond_curr_quiet     = curr_win_med_db <= global_mean_db - 2.0  # un po' sotto la media del brano

    #differenza tra curr_max_db e curr_min_db sopra una certa soglia
    #interpretazione: dentro questi 3 secondi ci sono stacchi / variazioni, non un muro piatto.
    cond_curr_has_gaps  = (curr_win_max_db - curr_win_min_db) >= 6.0   # c'è dinamica (stacchi / vuoti)

    # (b) futuro: esplosione di energia e basse forti
    cond_future_loud_enough = fut_win_max_db >= global_mean_db     # il futuro è forte rispetto al brano

    #nei prossimi 3 secondi c’è una grossa impennata di energia rispetto a ora (tipico del drop).
    cond_future_much_louder = (fut_win_max_db - curr_win_med_db) >= drop_boost_db

    #il picco delle low nel futuro è vicino al picco totale (fut_low_max_db ≈ fut_max_db)
    #interpretazione: non solo energia in generale, ma botta sulle basse (kick + bassline del drop).
    cond_future_low_strong  = (fut_win_low_max_db >= fut_win_max_db - 3.0)  # low vicine al picco totale

    if (cond_low_suppressed and cond_curr_quiet and cond_curr_has_gaps and
        cond_future_loud_enough and cond_future_much_louder and cond_future_low_strong):
        y = 1
        print(f"Trovato un drop al minuto {end_idx_curr_win * delta_frame}")
        input()
    else:
        y = 0

    print(f"Secondi mancati: {round((last_song_frame - start_idx_curr_win) * delta_frame)}")

    start_idx_curr_win += delta_between_curr_wins







'''
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


'''
