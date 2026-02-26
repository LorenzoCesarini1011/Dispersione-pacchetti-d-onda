import numpy as np

# Generazione delle frequenze
def gen_freq(N):
    nu = np.empty(0)
    pmax = 2/3
    while len(nu) < N:
        nucand = np.random.uniform(0,3)
        p_nugen = np.random.uniform(0,pmax)
        if nucand <= 2:
            p_nu = (1/3) * nucand
        else:
            p_nu = (2/3) * (3-nucand)
        if p_nugen < p_nu:
            nu = np.append(nu, nucand)
    return np.array(nu)

# Funzione per avere l'input di N
def chied_N():
    while True:
        
        try:
            N = int(input('Inserisci il numero di componenti N del pacchetto: '))
            if N < 100:
                    raise ValueError
            return N
        except ValueError:
                    print('Il numero deve essere un intero maggiore o uguale a 100')           

# Generazione delle ampiezze
def gen_amp(nu, a):
    Amax_nu = a / np.sqrt(nu)
    u = np.random.uniform(0,1,len(nu))
    A = Amax_nu * np.sqrt(u)
    return A

# Funzione per avere l'input di a
def chied_a():
    while True:
        try:
            a = float(input('Inserisci la costante a per determinare il valore di ampiezza massima: '))
            if a <= 0:
                    raise ValueError
            return a
        except ValueError:
                    print('La costante deve essere un numero reale positivo')

# Funzione per avere l'input di c
def chied_c():
    while True:
        try:
            c = float(input('Inserisci la costante c della relazione di dispersione: '))
            if c <= 0:
                    raise ValueError
            return c
        except ValueError:
                    print('La costante deve essere un numero reale positivo')

# Funzione per avere l'input di b
def chied_b(omega_min):
    while True:
        try:
            b = float(input('Inserisci la costante b della relazione di dispersione: '))
            if b <= 0:
                print("La costante deve essere maggiore di 0")
            elif b >= omega_min**2:
                print(f"La costante deve essere minore di omega_min^2 = {omega_min**2:.3f}")
            else:
                return b
        except ValueError:
            print('La costante deve essere un numero reale positivo accettabile')
