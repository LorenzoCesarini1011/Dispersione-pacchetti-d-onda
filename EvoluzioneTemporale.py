import numpy as np
import matplotlib.pyplot as plt

# Evoluzione temporale del pacchetto d'onda
def animazione_pacchetto(x, psi_xt, t_ciclo, pause_time):

    Nt, Nx = psi_xt.shape
    current_max = np.max(np.abs(psi_xt))

    fig, ax = plt.subplots(figsize = (15,6))
    linea, = ax.plot(x, psi_xt[0, :], color = 'blue')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-1.1*current_max, 1.1*current_max)
    ax.set_title("Evoluzione temporale del pacchetto d'onda")
    ax.set_xlabel('x')
    ax.set_ylabel('ψ(x,t)')
    ax.grid(True)

    # Testo del tempo in alto a sinistra
    time_text = ax.text(0.02, 0.95, '', transform = ax.transAxes,
                        fontsize = 14, verticalalignment = 'top',
                        bbox = dict(boxstyle = "round", facecolor = "white", alpha = 0.8))

    for it in range(0, Nt):
        t = t_ciclo[it]
        linea.set_ydata(psi_xt[it, :])
        time_text.set_text(f"t = {t:.2f} s")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause_time)

    # Mostra l'ultimo frame
    plt.show()

# Plot di un istante dell'evoluzione temporale
def plot_istante(x, psi_xt, it, title = None):
    current_max = np.max(np.abs(psi_xt))
    plt.figure(figsize = (15,6))
    plt.plot(x, psi_xt[it, :], color = 'blue', lw = 2)
    plt.ylim(-1.1*current_max, 1.1*current_max)
    plt.xlabel('x')
    plt.ylabel('ψ(x,t)')
    if title is not None:
        plt.title(title)
    plt.grid(True)
    plt.show()

# Funzione di analisi senza dispersione reale
def analisi_dispersione_reale(x, psi_xt, t_ciclo, window_smooth):
    sigma_t = []
    x_media_t = []

    for psi in psi_xt:
        peso = psi**2
        x_media = np.sum(x * peso) / np.sum(peso)
        sigma = np.sqrt(np.sum((x - x_media)**2 * peso) / np.sum(peso))

        x_media_t.append(x_media)
        sigma_t.append(sigma)

    sigma_t = np.array(sigma_t)
    x_media_t = np.array(x_media_t)

    # smoothing per velocità di gruppo
    x_media_smooth = np.convolve(x_media_t, np.ones(window_smooth) / window_smooth, mode =  'same')

    # fit lineare sul centro smussato
    coef = np.polyfit(t_ciclo, x_media_smooth, 1)
    v_gruppo = coef[0] 

    # controllo dispersione
    var_sigma = np.max(sigma_t) - np.min(sigma_t)
    if var_sigma / sigma_t[0] < 1e-3:
        print(f"Pacchetto non dispersivo: Δσ/σ_0 = {var_sigma/sigma_t[0]:.5f}")
    else:
        print(f"Pacchetto dispersivo: Δσ/σ_0 = {var_sigma/sigma_t[0]:.5f}")

    print(f"Velocità di gruppo del pacchetto reale: {v_gruppo:.3f} m/s")

    return sigma_t, x_media_t, v_gruppo

# Funzione di analisi della dispersione
def analisi_dispersione(x, psi_xt, t_ciclo, window_smooth):
    sigma_t = []
    x_media_t = []

    for psi in psi_xt:
        peso = np.abs(psi)**2
        x_media = np.sum(x * peso) / np.sum(peso)
        sigma = np.sqrt(np.sum((x - x_media)**2 * peso) / np.sum(peso))

        x_media_t.append(x_media)
        sigma_t.append(sigma)

    sigma_t = np.array(sigma_t)
    x_media_t = np.array(x_media_t)

    # smoothing per velocità istantanea
    x_media_smooth = np.convolve(x_media_t, np.ones(window_smooth) / window_smooth, mode = 'same')

    # velocità istantanea e velocità media del centro
    v_istantanea = np.gradient(x_media_smooth, t_ciclo)
    v_media = np.mean(v_istantanea)

    # controllo dispersione
    var_sigma = np.max(sigma_t) - np.min(sigma_t)
    disp_rel = var_sigma / sigma_t[0]

    if disp_rel < 1e-3:
        print(f"Pacchetto non dispersivo (Δσ/σ_0 = {disp_rel:.5f})")
    else:
        print(f"Pacchetto dispersivo (Δσ/σ_0 = {disp_rel:.5f})")

    # print(f"Velocità media del centro: v ≈ {v_media:.3f} m/s")
    print(f"Variazione massima della velocità istantanea: Δv ≈ {np.max(v_istantanea) - np.min(v_istantanea):.3e} m/s")

    return sigma_t, x_media_t, v_istantanea, v_media

# Plot della larghezza del pacchetto
def plot_larghezza(sigma_t, t_ciclo):
    plt.figure(figsize = (15,6))
    plt.plot(t_ciclo, sigma_t/sigma_t[0], 'b', lw = 2)
    plt.xlabel("tempo [s]")
    plt.ylabel("σ(t)/σ_0")
    plt.title("Larghezza relativa del pacchetto")
    plt.grid(True)
    plt.show()

# Plot della posizione del centro del pacchetto
def plot_x_media(x_media_t, t_ciclo):
    plt.figure(figsize = (15,6))
    plt.plot(t_ciclo, x_media_t, 'b', lw = 2)
    plt.xlabel("tempo [s]")
    plt.ylabel("x_media(t) [m]")
    plt.title("Centro del pacchetto")
    plt.grid(True)
    plt.show()

# Funzione di analisi della dispersione ideale
def analisi_dispersione_ideale(x, psi_xt_ideale, t_ciclo):
    x_media_ideale = []
    for psi in psi_xt_ideale:
        peso = psi**2
        x_media_ideale.append(np.sum(x*peso)/np.sum(peso))
    x_media_ideale = np.array(x_media_ideale)

    v_gruppo_ideale = np.polyfit(t_ciclo, x_media_ideale, 1)[0]
    print(f"Velocità di gruppo del pacchetto ideale: {v_gruppo_ideale:.3f} m/s")

    return x_media_ideale, v_gruppo_ideale

# Plot confronto centro pacchetto reale ed ideale
def plot_reale_ideale(x_media_t, x_media_ideale, t_ciclo):
    plt.figure(figsize = (15,6))
    plt.plot(t_ciclo, x_media_t, 'b', lw = 2, label = 'Pacchetto reale')
    plt.plot(t_ciclo, x_media_ideale, 'r--', lw = 2, label = 'Pacchetto ideale')
    plt.xlabel("tempo [s]")
    plt.ylabel("x_media(t) [m]")
    plt.title("Confronto centro pacchetto reale ed ideale")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot velocità istantanea del centro del pacchetto
def plot_v_istantanea(v_istantanea, t_ciclo):
    plt.figure(figsize = (15,6))
    plt.plot(t_ciclo, v_istantanea)
    plt.xlabel("tempo [s]")
    plt.ylabel("v_istantanea(t) [m/s]")
    plt.title("Velocità istantanea del centro del pacchetto")
    plt.grid(True)
    plt.show()

# Grafico confronto velocità di gruppo
def plot_v_confronto(v_gruppo_dom, v_gruppo_media_pesata0, v_gruppo_media_pesata5, x_media_t, t_ciclo):
    plt.figure(figsize = (15,6))
    plt.plot(t_ciclo, np.full_like(t_ciclo, v_gruppo_dom), '--r', label = 'Picco dominante')
    plt.plot(t_ciclo, np.full_like(t_ciclo, v_gruppo_media_pesata0), '--b', label = 'Media pesata spettro iniziale')
    plt.plot(t_ciclo, np.full_like(t_ciclo, v_gruppo_media_pesata5), '--c', label = 'Media pesata spettro finale')
    plt.plot(t_ciclo, np.gradient(x_media_t, t_ciclo), '-k', label = 'Velocità istantanea centro pacchetto')
    plt.xlabel("tempo [s]")
    plt.ylabel("v_gruppo [m/s]")
    plt.title("Confronto velocità di gruppo")
    plt.legend()
    plt.grid(True)
    plt.show()
