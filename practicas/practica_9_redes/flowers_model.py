import pandas as pd
import numpy as np
from constants import (
    Ydf_iris_sectosa,
    Ydf_iris_versicolor,
    Ydf_iris_virginica, mapeo_especies
)
from training_logic import (
    entrenar,
    normalizar_z_score,
    desnormalizar,
    guardar_modelo, loadModelFromDisk, predecir, escalonar
)

def entrenar_modelo_flores():
    df = pd.read_csv("IRIS.csv")
    X = df.iloc[:, 0:4].to_numpy()
    X_estadarizado, media, desviacion = normalizar_z_score(X)

    # sacamos un Yd real de las especies de los cada flor,
    # reucerda que es Yd por cada valor
    columna_especie = df.columns[-1]  # Toma la última columna


    Yd_completo = np.array([mapeo_especies[especie] for especie in df[columna_especie]])

    W_norm, B_norm, ECM = entrenar(X_estadarizado,Yd_completo,4,3,[128],0.1,5_000)
    W, B = desnormalizar(W_norm,B_norm,media,desviacion)
    guardar_modelo("flowers_model.npz",W,B)


def test_flower_model():
    W, B = loadModelFromDisk("flowers_model.npz")
    m = 4
    # 3. Crear x como una matriz de (1 fila, n_in columnas)
    x = np.zeros((1,m))

    # 4. Capturar datos
    for i in range(m):
        x[0,i] = float(input(f"Valor de característica {i + 1}: "))
    print("x: ", x)

    Yobt = predecir(x, W, B)[0]
    Yesc = escalonar(Yobt)

    print("Yobt: ", Yobt)
    print("Yobt (escalonada):",Yesc)


