from matriz import mapa # importamos el mapa desde un archivo aparte
import numpy as np
import matplotlib.pyplot as plt

movimientos = [(1,-1),(-1,-1),(-1,0),(1,0),(0,1),(1,1),(-1,1),(0,-1)]

TRANSITABLE = 0

NO_TRANSITABLE = 1

G_DIAGONAL = 14
G_CROSS = 10

ANIMATION_DELAY = 0.0001

class Nodo():
    def __init__(self,x, y, g = 0, f = 0, padre = None):
        self.x = x
        self.y = y
        self.g = g
        self.f = f
        self.padre = padre
        pass


    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def toPoint(self):
        return self.x, self.y

    def construir_camino(self):
        camino = []
        nodo_final = self
        while nodo_final is not None:
            camino.append(nodo_final)
            nodo_final = nodo_final.padre
        camino.reverse() # invertimos el camino, dado que esta desde final al primer
        return camino



def heuristica(nodo: Nodo,meta: Nodo):
        return abs(nodo.x - meta.x) + abs(nodo.y - meta.y)


def a_estrella(mapa,punto_inicial: Nodo, meta: Nodo) -> tuple[Nodo,list[Nodo]]:

    if mapa[punto_inicial.x,punto_inicial.y] == NO_TRANSITABLE:
        raise Exception('punto inicial no transitable')

    if mapa[meta.x,meta.y] == NO_TRANSITABLE:
        raise Exception('meta no transitable')

    lista_abierta = [
        Nodo(
            x=punto_inicial.x,
            y=punto_inicial.y,
            g=0,
            f=heuristica(punto_inicial,meta)
        )
    ]

    considerados = []

    filas,columnas = np.shape(mapa)

    lista_cerrada = np.zeros((filas,columnas))

    while len(lista_abierta) > 0:
        nodo_actual = lista_abierta[0] # sacamos el primer elemnto

        # recorremos la lista abierta para encontrar un f menor a nodo actual
        indice_menor_f = 0 # para saber el indice del nodo con menor f
        for i in range(1, len(lista_abierta)):
            nodo = lista_abierta[i]
            if nodo.f < nodo_actual.f:
                nodo_actual = lista_abierta[i]
                indice_menor_f = i
        # eliminamos de la lista abierta
        lista_abierta.pop(indice_menor_f) # sace y elimina el elemento en indice
        considerados.append(nodo_actual) #agregamos las coordeandas del nodo como considerado
        # preguntamos es la meta?
        if nodo_actual == meta:
            return nodo_actual, considerados

        # indicamos que el nodo fue utilizado
        lista_cerrada[nodo_actual.x,nodo_actual.y] = 1

        # verificamos los vecimos en base a los movimientos

        for vx,vy in movimientos:
            vecino = Nodo(x=nodo_actual.x + vx,y=nodo_actual.y + vy)
            # corroboramos si el vecino esta entre dentro de mapa y este sea valido
            if (0 <= vecino.x < filas  # que sea dentro de las filas
                    and 0 <= vecino.y < columnas # dentro de las columnas
                    and mapa[vecino.x,vecino.y] == TRANSITABLE # que sea transitable
                    and lista_cerrada[vecino.x,vecino.y] == 0): # que no ha sido vistado

                # evaluamos si es diagonal o horizontal
                if (abs(vx) + abs(vy)) == 2:
                    # implica que es diagonal
                    vecino.g = nodo_actual.g + G_DIAGONAL
                else: # de no ser el caso es un movimiento en vertical o horizontal
                    vecino.g = nodo_actual.g + G_CROSS

                vecino.f = vecino.g + heuristica(vecino,meta)

                # validamos si este vencino esta en lista para
                # si esta no esta, se agrega
                # si esta, y tiene un f menor que vecino, el nodo existente es mejor (menor costo)
                # no necesitamos a agregarlo
                # garantizamos nodos con con menores costos
                # sin embargo, arrastra nodos diplicados con costes menores
                # quiza, lo recomando seria quitar nodos duplicados para mejorar memoria
                bandera_lista = False

                for nodo in lista_abierta:
                    if nodo == vecino and nodo.f <= vecino.f:
                        bandera_lista = True
                        break

                if not bandera_lista:
                    vecino.padre = nodo_actual
                    lista_abierta.append(vecino)
    return None, considerados



def render(mapa,camino, considerados):
    plt.imshow(mapa, cmap='binary')

    if considerados:
        for nodo in considerados:
            plt.pause(ANIMATION_DELAY)
            plt.plot(nodo.y, nodo.x, 'o',color='blue')
    if camino:
        for nodo in camino:
            plt.pause(ANIMATION_DELAY)
            plt.plot(nodo.y, nodo.x, 'o',color='red')

    plt.show()



def main():
    punto_inicial = Nodo(1,1)
    meta = Nodo(47,47)

    punto_final, considerados = a_estrella(mapa,punto_inicial, meta)
    camino = punto_final.construir_camino()

    render(mapa,camino,considerados)
    pass


if __name__ == '__main__':
    main()