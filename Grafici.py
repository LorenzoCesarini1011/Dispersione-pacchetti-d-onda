import numpy as np
import matplotlib.pyplot as plt

# Istogramma delle frequenze non normalizzato e normalizzato
def plot_freq(nu):
    N = len(nu)
    v = np.linspace(0,3,500)
    p_v = np.where(v <= 2, (1/3)*v, (2/3)*(3-v))

    fig,axs = plt.subplots(1, 2, figsize = (15,6))

    axs[0].hist(nu, bins = int(np.sqrt(N)), range = (-0.1, 3.1), alpha = 0.8, color = 'royalblue', ec = 'lightblue', label = 'Istogramma non normalizzato', density = False)
    axs[0].set_xlim(0, 3)
    axs[0].set_title('Istogramma delle frequenze')
    axs[0].set_xlabel('ν [Hz]')
    axs[0].set_ylabel('Conteggi')
    axs[0].legend()

    axs[1].plot(v, p_v, 'r-', lw = 2, label = 'FDP teorica')
    axs[1].hist(nu, bins = int(np.sqrt(N)), range = (-0.1,3.1), alpha = 0.8, color = 'royalblue', ec = 'lightblue', label = 'Istogramma normalizzato', density = True)
    axs[1].set_xlim(0,3)
    axs[1].set_title('Frequenze normalizzate e FDP teorica')
    axs[1].set_xlabel('ν [Hz]')
    axs[1].set_ylabel('Densità di probabilità')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Istogramma delle ampiezze
def plot_amp(A):
    N = len(A)
    Amax = np.max(A)
    fig,axs = plt.subplots(figsize = (15,6))

    axs.hist(A, bins = int(np.sqrt(N)), range = (-0.1,Amax+0.1), alpha = 0.8, color = 'royalblue', ec = 'lightblue', density = False)
    axs.set_xlim(0, Amax)
    axs.set_title('Istogramma delle ampiezze')
    axs.set_xlabel('A [u.a.]')
    axs.set_ylabel('Conteggi')

    plt.show()
