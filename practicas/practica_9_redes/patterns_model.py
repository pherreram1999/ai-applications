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
    loadModelFromDisk,
    predecir,
    escalonar,
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
    pass

# para el grid
class GridDraw:
    def __init__(self, filas=30, columnas=30, tamaño_celda=20):
        self.filas = filas
        self.columnas = columnas
        self.tamaño_celda = tamaño_celda

        # Inicializamos la matriz con ceros (bitmap)
        self.matriz = np.zeros((filas, columnas), dtype=int)

        self.root = tk.Tk()
        self.root.title("Dibujador de Patrones 30x30")

        # Crear el lienzo (Canvas)
        self.canvas = tk.Canvas(
            self.root,
            width=columnas * tamaño_celda,
            height=filas * tamaño_celda,
            bg="white"
        )
        self.canvas.pack()

        # Dibujar las líneas de la grilla
        for i in range(filas + 1):
            self.canvas.create_line(0, i * tamaño_celda, columnas * tamaño_celda, i * tamaño_celda, fill="lightgray")
        for j in range(columnas + 1):
            self.canvas.create_line(j * tamaño_celda, 0, j * tamaño_celda, filas * tamaño_celda, fill="lightgray")

        # Eventos del mouse
        self.canvas.bind("<B1-Motion>", self.pintar)  # Arrastrar clic izquierdo
        self.canvas.bind("<Button-1>", self.pintar)  # Clic izquierdo único

        # Botones de control
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x")

        tk.Button(btn_frame, text="Limpiar", command=self.limpiar).pack(side="left", padx=10, pady=5)
        tk.Button(btn_frame, text="Clasificar", command=self.obtener_matriz).pack(side="right", padx=10, pady=5)

    def pintar(self, event):
        # Calcular en qué celda se hizo clic
        col = event.x // self.tamaño_celda
        fila = event.y // self.tamaño_celda

        if 0 <= fila < self.filas and 0 <= col < self.columnas:
            # Actualizar matriz
            self.matriz[fila, col] = 1
            # Pintar el cuadrado en el canvas
            x1 = col * self.tamaño_celda
            y1 = fila * self.tamaño_celda
            x2 = x1 + self.tamaño_celda
            y2 = y1 + self.tamaño_celda
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")

    def limpiar(self):
        self.matriz.fill(0)
        self.canvas.delete("all")
        # Redibujar grilla
        for i in range(self.filas + 1):
            self.canvas.create_line(0, i * self.tamaño_celda, self.columnas * self.tamaño_celda, i * self.tamaño_celda,
                                    fill="lightgray")
        for j in range(self.columnas + 1):
            self.canvas.create_line(j * self.tamaño_celda, 0, j * self.tamaño_celda, self.filas * self.tamaño_celda,
                                    fill="lightgray")

    def obtener_matriz(self):
        # clasificamos la matriz
        x = self.matriz.flatten()
        W, B = loadModelFromDisk()
        Yobt = escalonar(predecir(x,W,B))
        print(Yobt)

    def run(self):
        self.root.mainloop()
