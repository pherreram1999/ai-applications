import matplotlib.pyplot as plt
from matplotlib.pyplot import title

from patterns import *
import numpy as np
import argparse
import os

# lista de nuestros patrones
patrones_base = [
    patron_cruz,
    patron_cuadrado,
    patron_rombo,
    patron_estrella,
    patron_circulo,
    patron_triangulo
]

mapa_patrones = {
    "cruz": patron_cruz,
    "cuadrado": patron_cuadrado,
    "rombo": patron_rombo,
    "estrella": patron_estrella,
    "circulo": patron_circulo,
    "triangulo": patron_triangulo
}

NUMERO_COPIAS_POR_MUESTRA = 10

NUMERO_COPIAS_TEST_POR_MUESTRA = 3

Yd_cruz = [1, 0, 0, 0, 0, 0]
Yd_cuadrado = [0, 1, 0, 0, 0, 0]
Yd_rombo = [0, 0, 1, 0, 0, 0]
Yd_estrella = [0, 0, 0, 1, 0, 0]
Yd_circulo = [0, 0, 0, 0, 1, 0]
Yd_triangulo = [0, 0, 0, 0, 0, 1]

Yd_base = [
    Yd_cruz, # cruz
    Yd_cuadrado, # cuadrado
    Yd_rombo, # rombo
    Yd_estrella, # estrella
    Yd_circulo, # circulo
    Yd_triangulo, # triangulo
]


def sigmoide(z):
    return 1/(1+np.exp(-z))

def d_sigmoide_z(z):
    s = sigmoide(z)
    return s*(1-s)


def loadModelFromDisk():
    """
    Carga los pesos y las Bias de modelo guardados en disco (W, B)
    :return:
    """
    if not os.path.exists("model.npz"):
        raise Exception("model.npz no encontrado")
    with np.load("model.npz", allow_pickle=True) as data:
        W = data['W']
        B = data['B']
        return W, B



def esPrediccionValidaYd(Yobt):
    for Yd in Yd_base:
        if Yd == Yobt:
            return True
    return False

def entrenar(X, Yd, n_in,n_out, n_layers, lr, epoch_max):
    """
    Entrena el modelo segun los patrones dados
    :param X: matriz de caractericas de forma (muestras,caracteristicas)
    :param Yd: matriz de etiquetas, donde cada lista un clase , correspondiente al numero de muestras
    :param n_in: numero de neuronas entradas
    :param n_out: numero de neuronas de salida salidas
    :param n_layers: lista de numero de neuronas ocultas por capa
    :param lr: learning rate
    :param epoch_max: numero de epocas maximas
    :return: W (pesos),B (bias), ECM_historico
    """

    # contiene cuantas neuronas tiene cada capa
    dimensiones = [n_in] + n_layers + [n_out]

    numero_dimensiones = len(dimensiones)

    numero_capas = len(dimensiones)

    W = [] # almacena los pesos

    B = [] # almacena las bias

    # incializamos los pesos, recordar que los pesos se conectan con las capas siguientes
    # es por ello que la union entre es una matriz de neuronas de la capa actual por
    # las neuronas de la capa siguiente

    # se mulitplica por valores chicos para que la valores generados sean chicos y la suma no se muy grande en un principio

    for i in range(numero_dimensiones - 1):
        W.append(
            np.random.randn(dimensiones[i], dimensiones[i + 1]) * 0.1
        )
        # El bias el valor que nos ayuda incializar los valores de la capa destino
        B.append(
            np.random.randn(dimensiones[i + 1]) * 0.1
        )


    ECM = 10 # nuestro error (distancia de Yobt respecto a Yd (dato real) )

    ECM_historico = []

    epoch = 0

    while ECM > 0.0 and epoch <= epoch_max:

        suma_ecm = 0

        for p in range(len(X)): # recorremos los patrones

            # ===== Propagacion hacia adelante

            # guardamos los valores de activacion, lo devuelto por nuestra funcion
            # de activacion, en caso del primer indice son valores de entrada
            A = [X[p]]
            # guardamos los valores antes de activar que se usaran para la derivada
            Z = []

            for i in range(len(W)):
                # recordar que es cada valor de activacion por cada peso
                z_actual = np.dot(A[i], W[i]) + B[i]
                Z.append(z_actual)
                A.append(sigmoide(z_actual))
                pass

            # la ultima activacion es la salida de esperada de Y obtenida
            # en este caso Y_obt es una lista de las etiquetas esperadas
            Y_obt = A[-1]

            # guardamos los errores para poder propagar hacia atras
            deltas = [None] * len(W)

            # recordar que devuelve un arreglo de los errores (distancias)
            error = Yd[p] - Y_obt

            suma_ecm += np.sum(error**2)

            # la (Deseado - Obtenido) * derivada que conecta los errores entre capas
            deltas[-1] = error * d_sigmoide_z(Z[-1]) # recordar que vamos hacia atras, por eso empezamos desde el final

            # propagamos el error
            # reverse itera un lista desde el final (la invierte) regresando un iterador
            for i in reversed(range(len(deltas) - 1)):
                # (Delta siguiente * Pesos) * derivada de Z actual
                deltas[i] =  np.dot(deltas[i+1], W[i+1].T) * d_sigmoide_z(Z[i])
                pass

            # actualizamos pesos
            for i in range(len(W)):
                W[i] += lr * np.outer(A[i], deltas[i])
                B[i] += lr * deltas[i]

            pass
        pass
        ECM = 0.5 * (suma_ecm / len(X))
        ECM_historico.append(ECM)
        epoch += 1

    pass


    return W,B, ECM_historico



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

def escalonar(Z):
    activacion = []
    for a in Z:
        activacion.append(
            1 if a > 0.5 else 0
        )
    return activacion


def predecir(x, W, B):
    activacion = x
    for i in range(len(W)):
        # Z = (entrada * pesos) + bias
        z = np.dot(activacion, W[i]) + B[i]
        # La nueva activación es el resultado de la sigmoide
        activacion = sigmoide(z)

    return activacion  # Devuelve el vector de 6 salidas


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


    # creamos otra figura para poder comparar
    plt.figure(figsize=(4, 7), num="Patrones Base")

    i = 1
    for nombre_patron, patron in mapa_patrones.items():
        plt.subplot(len(patrones_base), 1, i)
        plt.imshow(patron, cmap="gray")
        plt.title(f"{nombre_patron}: {Yd_base[i-1]}")
        plt.axis("off")
        i = i + 1
        pass
    plt.tight_layout()  # quita espacios en blanco



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

    W, B, ECM = entrenar(X, Yd, num_caracteristica, 6, [128], 0.1, 1000)

    print("ECM: " + str(ECM[-1]))


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

    # guardamos los modelos en disco
    np.savez("model", W=np.array(W, dtype=object), B=np.array(B, dtype=object))
    print("El calculo de W y B se guardo en disco: models.npz")
    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e","--entrenar",action="store_true", help="entrenar modelo y guardar los pesos y bias")
    parser.add_argument("-t","--test",action="store_true", help="probar el modelo")

    args = parser.parse_args()

    if args.entrenar:
        print("==== Entrenando modelo ====")
        return entrenar_modelo()
    elif args.test:
        print("==== Probando modelo ====")
        return test_model()



    pass


if __name__ == '__main__':
    main()







