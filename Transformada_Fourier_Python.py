#-------------------------------------------------------------------------------
# Name:        Actividad 2 codigo 1
# Purpose:
#
# Author:      moises cruz cruz
# Universidad Ciudadana de Nuevo León
# Señales y Sistemas
# Created:     29/06/2025
# Copyright:   (c) moises 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import seaborn as sns

# Configurar estilo (ahora usando seaborn correctamente)
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]

def senoidal_simple():
    # Parámetros de la señal
    Fs = 1000  # Frecuencia de muestreo (Hz)
    T = 1/Fs   # Período de muestreo
    L = 1000   # Longitud de la señal
    t = np.arange(L)*T  # Vector de tiempo

    # Crear señal: suma de senoides de 50 Hz y 120 Hz
    f1, f2 = 50, 120
    signal = 0.7*np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

    # Transformada de Fourier
    yf = fft(signal)
    xf = fftfreq(L, T)[:L//2]  # Solo frecuencias positivas

    # Visualización
    plt.figure()

    # Dominio del tiempo
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Dominio del tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Dominio de la frecuencia
    plt.subplot(2, 1, 2)
    plt.plot(xf, 2/L * np.abs(yf[0:L//2]))
    plt.title('Dominio de la frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

senoidal_simple()

def senoidal_con_ruido():
    Fs = 1000
    T = 1/Fs
    L = 1000
    t = np.arange(L)*T

    # Señal limpia
    f_clean = 150
    clean_signal = 2 * np.sin(2*np.pi*f_clean*t)

    # Añadir ruido
    noise = 0.5 * np.random.normal(size=L)
    noisy_signal = clean_signal + noise

    # Transformada de Fourier
    yf_clean = fft(clean_signal)
    yf_noisy = fft(noisy_signal)
    xf = fftfreq(L, T)[:L//2]

    # Visualización
    plt.figure(figsize=(12, 8))

    # Dominio del tiempo
    plt.subplot(2, 2, 1)
    plt.plot(t, clean_signal)
    plt.title('Señal limpia - Dominio del tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, noisy_signal)
    plt.title('Señal con ruido - Dominio del tiempo')
    plt.xlabel('Tiempo [s]')
    plt.grid(True)

    # Dominio de la frecuencia
    plt.subplot(2, 2, 3)
    plt.plot(xf, 2/L * np.abs(yf_clean[0:L//2]))
    plt.title('Señal limpia - Dominio de la frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(xf, 2/L * np.abs(yf_noisy[0:L//2]))
    plt.title('Señal con ruido - Dominio de la frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

senoidal_con_ruido()

def pulso_cuadrado():
    Fs = 1000
    T = 1/Fs
    L = 1000
    t = np.arange(L)*T

    # Crear pulso cuadrado
    freq = 10  # Frecuencia del pulso (Hz)
    duty = 0.2  # Ciclo de trabajo
    square_wave = 0.5 * (np.modf(freq * t)[0] < duty) + 0.5

    # Transformada de Fourier
    yf = fft(square_wave)
    xf = fftfreq(L, T)[:L//2]

    # Visualización
    plt.figure()

    # Dominio del tiempo
    plt.subplot(2, 1, 1)
    plt.plot(t, square_wave)
    plt.title('Pulso cuadrado - Dominio del tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)

    # Dominio de la frecuencia
    plt.subplot(2, 1, 2)
    plt.plot(xf, 2/L * np.abs(yf[0:L//2]))
    plt.title('Pulso cuadrado - Dominio de la frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.xlim(0, 200)

    plt.tight_layout()
    plt.show()

pulso_cuadrado()
