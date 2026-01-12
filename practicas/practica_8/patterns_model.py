import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from constants import (
    patrones_base,
    mapa_patrones,
    Yd_base,
    NUMERO_COPIAS_POR_MUESTRA,
    NUMERO_COPIAS_TEST_POR_MUESTRA
)

from training_logic import (
    entrenar,
    entrenar_sin_backprop,
    entrenar_sin_backprop_capa_oculta_aleatoria,
    entrenar_con_mutacion,
    loadModelFromDisk,
    predecir,
    escalonar,
    guardar_modelo
)

def ruido(patron, porcentaje=0.1):
    r = patron.copy()
    filas, columnas = r.shape
    n = int(porcentaje * filas * columnas)

    for _ in range(n):
        i = np.random.randint(0, filas)
        j = np.random.randint(0, columnas)
        r[i, j] = 1 - r[i, j]
    return r

def generar_patrones_con_ruido(patron, no_copias, porcentaje=0.4):
    copias = []
    for _ in range(no_copias):
        copias.append(ruido(patron, porcentaje))
    return copias

def dibujar_patrones_base():
    # creamos otra figura para poder comparar
    plt.figure(figsize=(4, 7), num="Patrones Base")

    i = 1
    for nombre_patron, patron in mapa_patrones.items():
        plt.subplot(len(patrones_base), 1, i)
        plt.imshow(patron, cmap="gray")
        plt.title(f"{nombre_patron}: {Yd_base[i - 1]}")
        plt.axis("off")
        i = i + 1
        pass
    plt.tight_layout()  # quita espacios en blanco

def test_model():
    W, B = loadModelFromDisk()
    # creamos 3 por cada patron con ruido para comprobar el modelo
    i = 1
    plt.figure(figsize=(10, 8),num="Patrones con Ruido")

    numero_muestras = len(patrones_base)
    # Definimos el número de columnas para mayor claridad
    columnas = NUMERO_COPIAS_TEST_POR_MUESTRA + 1



    for nombre_patron, patron in mapa_patrones.items():
        patrones_con_ruido = generar_patrones_con_ruido(patron, NUMERO_COPIAS_TEST_POR_MUESTRA, .10)
        for patron_con_ruido in patrones_con_ruido:
            plt.subplot(numero_muestras, columnas, i)
            plt.imshow(patron_con_ruido, cmap="gray")
            plt.axis('off')

            x = patron_con_ruido.flatten()
            Yobt = escalonar(predecir(x, W, B))

            plt.title(f"Prediccion: \n {Yobt}", color='green', fontsize=8, pad=2)


            i = i + 1
            pass
        pass
    plt.tight_layout() # quita espacios en blanco


    dibujar_patrones_base()

    plt.show()

def entrenar_modelo():
    # para entrenar y pueda clasificar, vamos a generar ademas de los perfectos,
    # agregar con ruido para que logre inferir

    X = []
    Yd = []

    for p in range(len(patrones_base)):
        patrones_repetido_con_ruido = generar_patrones_con_ruido(patrones_base[p], NUMERO_COPIAS_POR_MUESTRA)
        X += [patrones_base[p].flatten()]
        Yd += [Yd_base[p]]
        for pr in patrones_repetido_con_ruido:
            X.append(pr.flatten())
            Yd.append(Yd_base[p])
            pass

    num_caracteristica = len(X[0])  # recuerda que toma el valor de un padron flatten de 30x30 = 900

    #W, B, ECM = entrenar(X, Yd, num_caracteristica, 6, [128], 0.1, 1000)
    #W, B, ECM = entrenar_sin_backprop(X, Yd, num_caracteristica, 6, [128], 0.1, 1000)
    #W, B, ECM = entrenar_sin_backprop_capa_oculta_aleatoria(X, Yd, num_caracteristica, 6, [1000], 0.1, 1000)
    W, B, ECM = entrenar_con_mutacion(X, Yd, num_caracteristica, 6, [128], 0.1, 1000)
    


    print("ECM: " + str(ECM[-1]))
    
    guardar_modelo("model.npz", W, B) #REVISAR
    
    # validamos si nuestor modeo predice los patrones con los que se entreno
    validados = 0

    for p in range(len(patrones_base)):
        x = patrones_base[p].flatten()
        Yobt = escalonar(predecir(x,W,B))
        print(Yobt,Yd_base[p])
        if Yobt == Yd_base[p]:
            validados += 1

    if validados != len(patrones_base):
        print("El modelo no logro predecir todos los patrones base !!!")
        return
    
