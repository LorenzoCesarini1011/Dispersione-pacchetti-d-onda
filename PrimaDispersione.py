# Studio della relazione di dispersione ω = sqrt(c*k)

import numpy as np
import matplotlib.pyplot as plt
from Distribuzioni import gen_freq, chied_N, gen_amp, chied_a, chied_c
from Grafici import plot_freq, plot_amp
from EvoluzioneTemporale import animazione_pacchetto, plot_istante, analisi_dispersione, plot_larghezza, plot_x_media, plot_v_istantanea, plot_v_confronto
from SpettroDiFourier import analisi_spettro, animazione_spettro, spettro_istante, kfft_dominante

# Parametri principali
N = chied_N() # numero di componenti del pacchetto
freq = gen_freq(N) # frequenze secondo distribuzione traccia
omega = 2*np.pi*freq
a = chied_a() # costante per ampiezza massima
amp = gen_amp(freq, a) # ampiezze secondo traccia
c = chied_c() # costante di dispersione positiva

# Istogrammi di frequenze e ampiezze
plot_freq(freq)
plot_amp(amp)

# Tempo e spazio
dt = 1 / (4*np.max(freq)) # passo temporale secondo Nyquist-Shannon
Nt = int(5 / dt)  # numero di istanti
t_ciclo = np.linspace(0, 5, Nt)  # vettore dei tempi
phi = np.random.uniform(0, 2*np.pi, N)  # vettore delle fasi casuali

k = omega**2 / c
sigma_0 = 2*np.pi / (np.max(k) - np.min(k)) # larghezza iniziale del pacchetto
k_dom = k[np.argmax(amp)]
v_dom = (1/2) * np.sqrt(c) / np.sqrt(k_dom)
Nx = 20000
L = 5*sigma_0 + v_dom*5 # dω/dk per ω = sqrt(c*k)
x = np.linspace(-L, L, Nx)

# Generazione del pacchetto d'onda reale
psi_xt = np.zeros((Nt, Nx), dtype = complex)
for it, t in enumerate(t_ciclo): # onda che si muove verso destra
    arg = k[:, None]*x - omega[:, None]*t + phi[:, None] 
    psi_xt[it, :] = np.sum(amp[:, None]*np.exp(1j*arg), axis = 0)

# Evoluzione temporale del pacchetto ed istanti iniziale e finale
animazione_pacchetto(x, psi_xt, t_ciclo, 0.1)
plot_istante(x, psi_xt, it = 0, title = "Pacchetto d'onda iniziale (t = 0 s)")
plot_istante(x, psi_xt, it = -1, title = "Pacchetto d'onda finale (t = 5 s)")

# Analisi del pacchetto con grafici
sigma_t, x_media_t, v_istantanea, v_media = analisi_dispersione(x, psi_xt, t_ciclo, 5)
plot_larghezza(sigma_t, t_ciclo)
plot_x_media(x_media_t, t_ciclo)
plot_v_istantanea(v_istantanea, t_ciclo)

# Spettro di potenza di Fourier in x al passare di t
kfft, ck_t, modck_quadro = analisi_spettro(psi_xt, x)
animazione_spettro(kfft, modck_quadro, t_ciclo, 0.1, 0.01)

# Plot dello spettro all'istante iniziale e confronto tra il k dominante teorico e quello determinato dalla FFT
spettro_istante(kfft, modck_quadro, t_ciclo, 0, 0.01)
modck_quadro0 = modck_quadro[0, :]
idx_prev = np.argmax(amp)
kdom = omega[idx_prev]**2 / c # k = ω**2 / c
print("Studio all'istante iniziale (t = 0 s):")
kfftmax0, modck_quadro_max0 = kfft_dominante(kfft, modck_quadro0, kdom)

# Plot dello spettro all'istante finale e confronto tra il k dominante teorico e quello determinato dalla FFT
spettro_istante(kfft, modck_quadro, t_ciclo, 5, 0.01)
modck_quadro5 = modck_quadro[-1, :]
print("Studio all'istante finale (t = 5 s):")
kfftmax5, modck_quadro_max5 = kfft_dominante(kfft, modck_quadro5, kdom)

# Velocità di gruppo per ogni k, del picco dominante, e della media pesata sullo spettro iniziale e finale
v_gruppo_k = (1/2) * np.sqrt(c) / np.sqrt(kfft) 
v_gruppo_dom = (1/2) * np.sqrt(c) / np.sqrt(kdom)
v_gruppo_media_pesata0 = np.sum(modck_quadro0 * v_gruppo_k) / np.sum(modck_quadro0)
v_gruppo_media_pesata5 = np.sum(modck_quadro5 * v_gruppo_k) / np.sum(modck_quadro5)
print(f"Velocità di gruppo del picco dominante: {v_gruppo_dom:.3f} m/s")
print(f"Velocità di gruppo media pesata sullo spettro iniziale: {v_gruppo_media_pesata0:.3f} m/s")
print(f"Velocità di gruppo media pesata sullo spettro finale: {v_gruppo_media_pesata5:.3f} m/s")
print(f"Velocità di gruppo media del centro del pacchetto: {v_media:.3f} m/s")
diff_dom = np.abs(v_gruppo_dom - v_media)
diff_media0 = np.abs(v_gruppo_media_pesata0 - v_media)
diff_media5 = np.abs(v_gruppo_media_pesata5 - v_media)
print(f"Differenza picco dominante e centro: {diff_dom:.2e}")
print(f"Differenza media pesata iniziale e centro: {diff_media0:.2e}")
print(f"Differenza media pesata finale e centro: {diff_media5:.2e}")

plot_v_confronto(v_gruppo_dom, v_gruppo_media_pesata0, v_gruppo_media_pesata5, x_media_t, t_ciclo)
