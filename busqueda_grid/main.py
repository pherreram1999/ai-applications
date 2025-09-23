import numpy as np
import matplotlib.pyplot as plt

# Laberinto donde 1 = pared, 0 = transitable
mapa = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Punto de partida en la esquina superior izquierda (área transitable)
punto_inicial = (1, 1)  # Primera posición transitable después del borde

# Meta en la parte inferior derecha
meta = (1, 4)  # Posición transitable cerca de la esquina inferior derecha


# Reglas del agente
movimientos = [(0,1),(1,0),(0,-1),(-1,0)]


def dfs(mapa,punto_inicial,meta):
    pila = [(punto_inicial,[])] # para validar el camino
    #podemos crear un mapa de mismas dimensiones o copia del mapa
    # para almacenar si ha sido visitado
    filas, columnas = np.shape(mapa) # recumeramos la longitud de filas y columnas
    visitados = np.zeros((filas,columnas)) # creamo un vector cero
    # almacenar todos las coordenadas de los nodos que he visitado
    # y ademas para obtener las cordenadas del camino de la meta al punto
    # inicial
    considerados = []


    while len(pila) > 0:  # validar si aun tenemos camino
        nodo_actual, camino = pila[-1]
        pila = pila[:-1]
        # guardar el inicial en los considerados
        considerados += [nodo_actual]

        # nodo actual es la solucion?
        if nodo_actual == meta:
            return camino + [nodo_actual] , considerados

        # ya visitamos el nodo actual
        visitados[nodo_actual[0],nodo_actual[1]] = 1
        # recorrer nodos hijos
        for mov in movimientos:
            hijo = (nodo_actual[0]+ mov[0],nodo_actual[1] + mov[1]) # sumanos las posiciones de nodo actual encontrar vecinos
            print(hijo)
            # evaluamos que este dentro del mapa
            # que no halla sido visitado
            # que sea transitable ( aqui zero es transitable
            if ((0  <= hijo[0] < filas) and (0 <= hijo[1] < columnas)
                    and visitados[hijo[0],hijo[1]] == 0
                    and mapa[hijo[0],hijo[1]] == 0):
                # si se cumple. concatenas en la fila
                pila += [(hijo, camino + [nodo_actual])]
    return None, considerados # recorrio todos y no encontro ninguna solucion


def desplegar_laberinto(mapa, camino, considerados):
    plt.figure()
    plt.imshow(mapa, cmap='binary') # pinta los unos y ceros como blanco y negro
    plt.title('Mapa')

    if considerados: # primero pinto lo considerados
        for i in considerados:
            plt.plot(i[1], i[0], 'o', color='blue')

    if camino: # despues sobreescribo el camino
        for i in camino:
            plt.plot(i[1], i[0], 'o', color='red')

    plt.show()


def main():
    camino, considerados = dfs(mapa,punto_inicial,meta)
    desplegar_laberinto(mapa,camino,considerados)
    pass

if __name__ == '__main__':
    main()