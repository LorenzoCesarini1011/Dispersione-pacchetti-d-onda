# Studio della relazione di dispersione ω = c*k

import numpy as np
import matplotlib.pyplot as plt
from Distribuzioni import gen_freq, chied_N, gen_amp, chied_a, chied_c
from Grafici import plot_freq, plot_amp
from EvoluzioneTemporale import animazione_pacchetto, plot_istante, analisi_dispersione_reale, plot_larghezza, plot_x_media, analisi_dispersione_ideale, plot_reale_ideale
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
print(t_ciclo)
phi = np.random.uniform(0, 2*np.pi, N)  # vettore delle fasi casuali

k = omega / c
sigma_0 = 2*np.pi / (np.max(k) - np.min(k))  # larghezza iniziale del pacchetto
Nx = 20000
L = 5*sigma_0 + c*5 # dω/dk per ω = c*k
x = np.linspace(-L, L, Nx)

# Generazione del pacchetto d'onda reale
psi_xt = np.zeros((Nt, Nx))
for it, t in enumerate(t_ciclo): # onda che si muove verso destra
    arg = k[:, None]*x - omega[:, None]*t + phi[:, None] 
    psi_xt[it, :] = np.sum(amp[:, None]*np.cos(arg), axis = 0)

# Evoluzione temporale del pacchetto ed istanti iniziale e finale
animazione_pacchetto(x, psi_xt, t_ciclo, 0.1)
plot_istante(x, psi_xt, it = 0, title = "Pacchetto d'onda iniziale (t = 0 s)")
plot_istante(x, psi_xt, it = -1, title = "Pacchetto d'onda finale (t = 5 s)")

# Analisi del pacchetto reale con grafici
sigma_t, x_media_t, v_gruppo = analisi_dispersione_reale(x, psi_xt, t_ciclo, 5)
plot_larghezza(sigma_t, t_ciclo)
plot_x_media(x_media_t, t_ciclo)

# Confronto con il pacchetto d'onda ideale
phi_ideale = np.zeros(N) # tutte le fasi allineate
amp_ideale = np.ones(N) # ampiezze uniformi

psi_xt_ideale = np.zeros((Nt, Nx))
for it, t in enumerate(t_ciclo):
    arg = k[:, None]*x - omega[:, None]*t + phi_ideale[:, None]
    psi_xt_ideale[it, :] = np.sum(amp_ideale[:, None]*np.cos(arg), axis = 0)

x_media_ideale, v_gruppo_ideale = analisi_dispersione_ideale(x, psi_xt_ideale, t_ciclo)   

plot_reale_ideale(x_media_t, x_media_ideale, t_ciclo)

# Spettro di potenza di Fourier in x al passare di t
kfft, ck_t, modck_quadro = analisi_spettro(psi_xt, x)
animazione_spettro(kfft, modck_quadro, t_ciclo, 0.1, 0.01)

# Plot dello spettro all'istante iniziale e confronto tra il k dominante teorico e quello determinato dalla FFT
spettro_istante(kfft, modck_quadro, t_ciclo, 0, 0.01)
modck_quadro0 = modck_quadro[0, :]
idx_prev = np.argmax(amp)
kdom = k[idx_prev] # funzione lineare: picco ampiezza = picco FFT
print("Studio all'istante iniziale (t = 0 s):")
kfftmax0, modck_quadro_max0 = kfft_dominante(kfft, modck_quadro0, kdom)

# Plot dello spettro all'istante finale e confronto tra il k dominante teorico e quello determinato dalla FFT
spettro_istante(kfft, modck_quadro, t_ciclo, 5, 0.01)
modck_quadro5 = modck_quadro[-1, :]
print("Studio all'istante finale (t = 5 s):")
kfftmax5, modck_quadro_max5 = kfft_dominante(kfft, modck_quadro5, kdom)


