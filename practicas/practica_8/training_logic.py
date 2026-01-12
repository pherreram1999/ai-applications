import numpy as np
import os
from constants import Yd_base

def sigmoide(z):
    return 1/(1+np.exp(-z))

def d_sigmoide_z(z):
    s = sigmoide(z)
    return s*(1-s)


def loadModelFromDisk(model_name="model.npz"):
    if not os.path.exists(model_name):  # Asegúrate de usar el nombre que pasas por parámetro
        raise Exception(f"{model_name} no encontrado")

    with np.load(model_name, allow_pickle=True) as data:
        # Convertimos cada elemento de la lista 'W' y 'B' en un array de numpy numérico
        W = [np.array(w, dtype=float) for w in data['W']]
        B = [np.array(b, dtype=float) for b in data['B']]
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
            A = [X[p]] #primer entrada como valores de activacion
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
    
    
def entrenar_sin_backprop(X, Yd, n_in,n_out, n_layers, lr, epoch_max):
    
    dimensiones = [n_in] + n_layers + [n_out]
    numero_dimensiones = len(dimensiones)
    numero_capas = len(dimensiones)
    W = [] 
    B = [] 

    for i in range(numero_dimensiones - 1):
        W.append(np.random.randn(dimensiones[i], dimensiones[i + 1]) * 0.1)
        B.append(np.random.randn(dimensiones[i + 1]) * 0.1)

    ECM = 10 
    ECM_historico = []
    epoch = 0

    while ECM > 0.0 and epoch <= epoch_max:
        suma_ecm = 0
        for p in range(len(X)): 
            # === FORWARD PROPAGATION ===
            A = [X[p]]
            Z = []

            for i in range(len(W)):
                z_actual = np.dot(A[i], W[i]) + B[i]
                Z.append(z_actual)
                A.append(sigmoide(z_actual))

            Y_obt = A[-1]
            deltas = [None] * len(W)

            error = Yd[p] - Y_obt
            suma_ecm += np.sum(error**2)

            deltas[-1] = error * d_sigmoide_z(Z[-1]) # recordar que vamos hacia atras, por eso empezamos desde el final
            error_global_simple = np.sum(error)
            
            for i in reversed(range(len(deltas) - 1)):
                deltas[i] = error_global_simple * d_sigmoide_z(Z[i])   #SIN BACKPROPAGATION :(

            # actualizamos pesos
            for i in range(len(W)):
                W[i] += lr * np.outer(A[i], deltas[i])
                B[i] += lr * deltas[i]

        ECM = 0.5 * (suma_ecm / len(X))
        ECM_historico.append(ECM)
        epoch += 1

    return W,B, ECM_historico


    
    
def normalizar_z_score(X):
    """
    Normaliza una matriz de datos usando Z-Score (estandarización).
    :param X: Matriz de datos (muestras, caracteristicas) como np.array
    :return: Matriz normalizada
    """
    # Convertir a array de numpy por si se recibe una lista de listas
    X = np.array(X)

    # Calcular la media por cada columna (axis=0)
    media = np.mean(X, axis=0)

    # Calcular la desviación estándar por cada columna (axis=0)
    desviacion = np.std(X, axis=0)

    # Evitar división por cero en caso de que una característica sea constante
    desviacion[desviacion == 0] = 1.0

    # Aplicar la fórmula: (X - media) / desviacion
    X_norm = (X - media) / desviacion

    return X_norm,media,desviacion

def desnormalizar(W, B, media, desviacion):
    """
    Ajusta los pesos y bias de la primera capa para aceptar datos sin normalizar.
    :param W: Lista de matrices de pesos (W[0] es la entrada)
    :param B: Lista de vectores de bias (B[0] es la entrada)
    :param media: Vector de medias usado en Z-score
    :param desviacion: Vector de desviaciones usado en Z-score
    :return: Copias de W y B ajustadas
    """
    # Creamos copias para no modificar los originales
    W_real = [w.copy() for w in W]
    B_real = [b.copy() for b in B]

    # Ajustamos solo la PRIMERA capa (índice 0)
    # W_real[0] tiene forma (caracteristicas, neuronas_ocultas)
    # media y desviacion tienen forma (caracteristicas,)

    # 1. Ajustar Pesos: W / sigma (columna por columna)
    for i in range(W_real[0].shape[1]):  # Por cada neurona oculta
        W_real[0][:, i] = W_real[0][:, i] / desviacion

    # 2. Ajustar Bias: B - sum(mu * W / sigma)
    # Calculamos el ajuste restando el producto punto de la media con los nuevos pesos
    ajuste_bias = np.dot(media, W_real[0])
    B_real[0] = B_real[0] - ajuste_bias

    return W_real, B_real

def predecir(x, W, B):
    activacion = x
    for i in range(len(W)):
        # Z = (entrada * pesos) + bias
        z = np.dot(activacion, W[i]) + B[i]
        # La nueva activación es el resultado de la sigmoide
        activacion = sigmoide(z)

    return activacion  # Devuelve el vector de 6 salidas

def guardar_modelo(nombre_modelo,W,B):
    # guardamos los modelos en disco
    np.savez(nombre_modelo, W=np.array(W, dtype=object), B=np.array(B, dtype=object))
    print("El calculo de W y B se guardo en disco: " + nombre_modelo + " !!!")

def escalonar(Z):
    activacion = []
    for a in Z:
        activacion.append(
            1 if a > 0.5 else 0
        )
    return activacion
