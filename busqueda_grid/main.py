import numpy as np
# Posicion Inicial
punto_inicial = (1,1)
# meta
meta = (13,14)

# Reglas del agente
movimientos = [(0,1),(1,0),(0,-1),(-1,0)]


def dfs(mapa,punto_inicial,meta):
    pila = [(punto_inicial),[]] # para validar el camino
    #podemos crear un mapa de mismas dimensiones o copia del mapa
    # para almacenar si ha sido visitado
    fila, columnas = np.shape(mapa) # recumeramos la longitud de filas y columnas
    visitados = np.zeros((fila,columnas)) # creamo un vector cero
    # almacenar todos las coordenadas de los nodos que he visitado
    # y ademas para obtener las cordenadas del camino de la meta al punto
    # inicial
    considerados = []


    while len(pila) > 0:
        nodo_actual, camino = pila[-1]
        pila = pila[:-1]
        # guardar el inicial en los considerados
        considerados += [nodo_actual]

        # nodo actual es la solucion?
        if nodo_actual == meta:
            return camino + [nodo_actual] , considerados

        # ya visitamos el nodo actual
        x,y = nodo_actual
        visitados[x,y] = 1
        # recorrer nodos hijos
        for hx,hy in movimientos:
            hijo = (x+hx,hy) # sumanos las posiciones de nodo actual encontrar vecinos
