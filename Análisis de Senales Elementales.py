#-------------------------------------------------------------------------------
# Name:        Actividad 2 codigo 2
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
from scipy import signal as sp

# Configuración general
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.grid'] = True

# =============================================
# 1. Definición de funciones para generar señales
# =============================================

def pulso_rectangular(t, ancho=1, amplitud=1):
    """
    Genera un pulso rectangular centrado en t=0

    Parámetros:
    t -- array de tiempos
    ancho -- duración del pulso (en segundos)
    amplitud -- amplitud del pulso

    Retorna:
    señal -- array con el pulso rectangular
    """
    return amplitud * (np.abs(t) <= ancho/2).astype(float)

def funcion_escalon(t, t0=0, amplitud=1):
    """
    Genera una función escalón unitario

    Parámetros:
    t -- array de tiempos
    t0 -- instante de transición
    amplitud -- amplitud del escalón

    Retorna:
    señal -- array con la función escalón
    """
    return amplitud * (t >= t0).astype(float)

def senoidal(t, frecuencia=1, amplitud=1, fase=0):
    """
    Genera una señal senoidal

    Parámetros:
    t -- array de tiempos
    frecuencia -- frecuencia en Hz
    amplitud -- amplitud de la señal
    fase -- fase inicial en radianes

    Retorna:
    señal -- array con la señal senoidal
    """
    return amplitud * np.sin(2*np.pi*frecuencia*t + fase)

# =============================================
# 2. Configuración común para todas las señales
# =============================================

# Parámetros de muestreo
Fs = 1000  # Frecuencia de muestreo (Hz)
T = 1/Fs   # Período de muestreo (s)
L = 2000   # Número de muestras
t = np.arange(-L/2, L/2)*T  # Vector de tiempo centrado en 0

# =============================================
# 3. Generación y análisis de cada señal
# =============================================

def analizar_senal(senal, nombre, t=t):
    """
    Función que realiza el análisis completo de una señal:
    - Grafica la señal en tiempo
    - Calcula la FFT
    - Grafica magnitud y fase
    - Verifica propiedades

    Parámetros:
    senal -- array con la señal a analizar
    nombre -- nombre para los títulos de gráficos
    t -- array de tiempos
    """

    # Asegurarse que la señal tenga número impar de muestras para FFT
    if len(senal) % 2 == 0:
        senal = np.append(senal, 0)
        t = np.append(t, t[-1]+T)

    # ----------------------------
    # Cálculo de la FFT
    # ----------------------------
    N = len(senal)
    yf = np.fft.fft(senal)
    xf = np.fft.fftfreq(N, T)
    xf = np.fft.fftshift(xf)  # Reordenar frecuencias
    yf = np.fft.fftshift(yf)  # Reordenar FFT

    # Magnitud y fase
    magnitud = np.abs(yf) / N
    fase = np.angle(yf)

    # ----------------------------
    # Visualización
    # ----------------------------
    plt.figure(figsize=(14, 10))

    # Señal en tiempo
    plt.subplot(3, 1, 1)
    plt.plot(t, senal)
    plt.title(f'{nombre} - Dominio del tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')

    # Magnitud en frecuencia
    plt.subplot(3, 1, 2)
    plt.plot(xf, magnitud)
    plt.title('Espectro de magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.xlim([-50, 50])  # Limitar rango para mejor visualización

    # Fase en frecuencia
    plt.subplot(3, 1, 3)
    plt.plot(xf, fase)
    plt.title('Espectro de fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.xlim([-50, 50])

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Verificación de propiedades
    # ----------------------------
    print(f"\nAnálisis de propiedades para {nombre}:")

    # Linealidad
    senal1 = senoidal(t, frecuencia=5)
    senal2 = senoidal(t, frecuencia=10)
    combo = senal1 + senal2

    yf1 = np.fft.fft(senal1)
    yf2 = np.fft.fft(senal2)
    yf_combo = np.fft.fft(combo)

    error_linealidad = np.max(np.abs(yf_combo - (yf1 + yf2)))
    print(f"Error de linealidad: {error_linealidad:.2e} (debe ser cercano a 0)")

    # Desplazamiento en tiempo
    desplazamiento = 0.1  # segundos
    senal_desplazada = np.roll(senal, int(desplazamiento/T))
    yf_desplazada = np.fft.fft(senal_desplazada)

    # Teoría: FFT(s(t-t0)) = FFT(s(t)) * exp(-j*2*pi*f*t0)
    factor_teorico = np.exp(-2j*np.pi*xf*desplazamiento)
    error_desplazamiento = np.max(np.abs(yf_desplazada - yf*factor_teorico))
    print(f"Error en desplazamiento temporal: {error_desplazamiento:.2e}")

    # Escalamiento en frecuencia
    factor_escala = 2
    senal_escalada = senal[::factor_escala]
    t_escalado = t[::factor_escala]

    if len(senal_escalada) % 2 == 0:
        senal_escalada = np.append(senal_escalada, 0)
        t_escalado = np.append(t_escalado, t_escalado[-1]+T)

    yf_escalada = np.fft.fft(senal_escalada)
    xf_escalada = np.fft.fftfreq(len(senal_escalada), T*factor_escala)
    xf_escalada = np.fft.fftshift(xf_escalada)
    yf_escalada = np.fft.fftshift(yf_escalada)

    # Teoría: Comprimir en tiempo expande en frecuencia
    print("Verificación visual de escalamiento en frecuencia (comparar gráficos)")

    return xf, magnitud, fase

# =============================================
# 4. Generación y análisis de cada señal
# =============================================

# Pulso rectangular
print("\n" + "="*60)
print("ANÁLISIS DEL PULSO RECTANGULAR")
print("="*60)
pulso = pulso_rectangular(t, ancho=0.2, amplitud=1)
xf_pulso, mag_pulso, fase_pulso = analizar_senal(pulso, "Pulso rectangular")

# Función escalón
print("\n" + "="*60)
print("ANÁLISIS DE LA FUNCIÓN ESCALÓN")
print("="*60)
escalon = funcion_escalon(t, t0=0, amplitud=1)
xf_escalon, mag_escalon, fase_escalon = analizar_senal(escalon, "Función escalón")

# Señal senoidal
print("\n" + "="*60)
print("ANÁLISIS DE LA SEÑAL SENOIDAL")
print("="*60)
seno = senoidal(t, frecuencia=10, amplitud=1, fase=0)
xf_seno, mag_seno, fase_seno = analizar_senal(seno, "Señal senoidal")

# =============================================
# 5. Comparación entre señales
# =============================================

plt.figure(figsize=(14, 8))

# Comparación de espectros de magnitud
plt.subplot(2, 1, 1)
plt.plot(xf_pulso, mag_pulso, label='Pulso rectangular')
plt.plot(xf_escalon, mag_escalon, label='Función escalón')
plt.plot(xf_seno, mag_seno, label='Señal senoidal')
plt.title('Comparación de espectros de magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.xlim([-20, 20])
plt.legend()

# Comparación de espectros de fase
plt.subplot(2, 1, 2)
plt.plot(xf_pulso, fase_pulso, label='Pulso rectangular')
plt.plot(xf_escalon, fase_escalon, label='Función escalón')
plt.plot(xf_seno, fase_seno, label='Señal senoidal')
plt.title('Comparación de espectros de fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim([-20, 20])
plt.legend()

plt.tight_layout()
plt.show()
