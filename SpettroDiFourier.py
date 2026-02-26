import numpy as np
import matplotlib.pyplot as plt

# Analisi FFT e spettro di potenza (solo k > 0)
def analisi_spettro(psi_xt, x):

    dx = x[1] - x[0]
    Nx = len(x)

    # FFT completa lungo l'asse spaziale
    ck = np.fft.fft(psi_xt, axis = 1)
    k = np.fft.fftfreq(Nx, d = dx)

    # Consideriamo solo k > 0
    mask = k > 0
    kfft = k[mask]
    ck_t = ck[:, mask]
    modck_quadro = np.abs(ck_t)**2

    return kfft, ck_t, modck_quadro

# Animazione spettro (solo k > 0)
def animazione_spettro(kfft, modck_quadro, t_ciclo, pause_time, soglia_frac):

    Nt, Nx = modck_quadro.shape
    fig, ax = plt.subplots(figsize = (15,6))

    linea, = ax.plot(kfft, modck_quadro[0, :], color = 'b')
    ax.set_xlabel('kFFT')
    ax.set_ylabel(r'$|c_k|^2$')
    ax.set_title("Evoluzione temporale dello spettro di potenza")
    ax.grid(True)

    time_text = ax.text(0.02, 0.95, '', transform = ax.transAxes,
                        fontsize = 14, verticalalignment = 'top',
                        bbox = dict(boxstyle = "round", facecolor = "white", alpha  = 0.8))

    for it in range(0, Nt):
        modck = modck_quadro[it, :]
        linea.set_ydata(modck)
        ax.set_ylim(0, 1.1*np.max(modck))

        # zoom su valori significativi
        mask_zoom = modck > soglia_frac * np.max(modck)
        if np.any(mask_zoom):
            k_min, k_max = kfft[mask_zoom].min(), kfft[mask_zoom].max()
            delta = 0.05 * (k_max - k_min)
            ax.set_xlim(k_min - delta, k_max + delta)

        time_text.set_text(f"t = {t_ciclo[it]:.2f} s")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause_time)

    plt.show()

# Spettro ad un istante specifico (solo k > 0)
def spettro_istante(kfft, modck_quadro, t_ciclo, t_scelto, soglia_frac):
    
    idx = np.argmin(np.abs(t_ciclo - t_scelto))
    modck_quadro = modck_quadro[idx, :]

    plt.figure(figsize = (15,6))
    plt.plot(kfft, modck_quadro, color = 'b')
    plt.xlabel('kFFT')
    plt.ylabel(r'$|c_k|^2$')
    plt.title(f"Spettro di potenza all'istante t = {t_ciclo[idx]:.2f} s")
    plt.grid(True)

    mask_zoom = modck_quadro > soglia_frac * np.max(modck_quadro)
    if np.any(mask_zoom):
        k_min, k_max = kfft[mask_zoom].min(), kfft[mask_zoom].max()
        delta = 0.05 * (k_max - k_min)
        plt.xlim(k_min - delta, k_max + delta)

    plt.show()

# Numero d'onda dominante dall'FFT (solo k > 0) con confronto al numero d'onda dominante previsto
def kfft_dominante(kfft, modck_quadro, kdom):
    # Trova il massimo dello spettro
    idx_max = np.argmax(modck_quadro)
    k_max = kfft[idx_max]
    modck_quadro_max = modck_quadro[idx_max]

    diff = np.abs(k_max - kdom)

    print(f"k dominante FFT: {k_max:.6f}")
    print(f"|c_k|^2 max: {modck_quadro_max:.3e}")
    print(f"k previsto: {kdom:.6f}")
    print(f"Differenza: {diff:.2e}\n")

    return k_max, modck_quadro_max
