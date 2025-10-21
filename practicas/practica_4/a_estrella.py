import math

from nodo import Nodo
from constants import NO_TRANSITABLE, TRANSITABLE, COSTOS,AGUA,LODO
import numpy as np

movimientos_estrella = [(1,-1),(-1,-1),(-1,0),(1,0),(0,1),(1,1),(-1,1),(0,-1)]

def heuristica(nodo: Nodo ,meta: Nodo):
    """
    TODO tenemos que considerar si es bien aplicado para nuestros costos
    Aplica el calculo de la heuristica de estrella usando Chebyshev L∞
    :param nodo:
    :param meta:
    :return:
    """
    return max(abs(meta.x - nodo.x), abs(meta.y - nodo.y))

def a_estrella(mapa ,punto_inicial: Nodo, meta: Nodo) -> tuple[Nodo ,list[Nodo]]:

    if mapa[punto_inicial.x ,punto_inicial.y] == NO_TRANSITABLE:
        raise Exception('punto inicial no transitable')

    if mapa[meta.x ,meta.y] == NO_TRANSITABLE:
        raise Exception('meta no transitable')

    lista_abierta = [
        Nodo(
            x=punto_inicial.x,
            y=punto_inicial.y,
            g=0,
            f=heuristica(punto_inicial ,meta)
        )
    ]

    considerados = []

    filas ,columnas = np.shape(mapa)

    lista_cerrada = np.zeros((filas ,columnas))

    while len(lista_abierta) > 0:
        nodo_actual = lista_abierta[0] # sacamos el primer elemento

        # recorremos la lista abierta para encontrar un f menor a nodo actual
        indice_menor_f = 0 # para saber el indice del nodo con menor f
        for i in range(1, len(lista_abierta)):
            nodo = lista_abierta[i]
            if nodo.f < nodo_actual.f:
                nodo_actual = lista_abierta[i]
                indice_menor_f = i
        # eliminamos de la lista abierta
        lista_abierta.pop(indice_menor_f) # saca y elimina el elemento en indice dado
        considerados.append(nodo_actual) # agregamos las coordeandas del nodo como considerado
        # preguntamos es la meta?
        if nodo_actual == meta:
            return nodo_actual, considerados

        # indicamos que el nodo fue utilizado
        lista_cerrada[nodo_actual.x ,nodo_actual.y] = 1

        # verificamos los vecimos en base a los movimientos

        for vx ,vy in movimientos_estrella:
            vecino = Nodo(x=nodo_actual.x + vx ,y=nodo_actual.y + vy)
            # corroboramos si el vecino esta entre dentro de mapa y este sea valido
            if (0 <= vecino.x < filas  # que sea dentro de las filas
                    and 0 <= vecino.y < columnas # dentro de las columnas
                    and mapa[vecino.x ,vecino.y] in [TRANSITABLE,AGUA,LODO] # que sea transitable
                    and lista_cerrada[vecino.x ,vecino.y] == 0): # que no ha sido vistado

                # determinamos el costo de g
                cx, cy = COSTOS[mapa[vecino.x ,vecino.y]]
                # evaluamos si es diagonal o horizontal
                if (abs(vx) + abs(vy)) == 2:
                    # implica que es diagonal
                    vecino.g = nodo_actual.g + math.sqrt(cx*cx + cy*cy)
                    # se hace el calculo segun si es vertical o diagonal
                elif abs(vx) == 1 and vy == 0: # se mueve a la derecha o izqueda (horizontal)
                    vecino.g = nodo_actual.g + cx
                elif abs(vy) == 1 and vx == 0: # se mueve arriba a obajo (vertical)
                    vecino.g = nodo_actual.g + cy


                vecino.f = vecino.g + heuristica(vecino ,meta)

                # validamos si este vencino esta en lista abierta para
                # si no esta, se agrega
                # si esta, y este tiene un f menor que vecino, el nodo existente es mejor (menor costo)
                # no necesitamos a agregarlo
                # garantizamos nodos con con menores costos
                # sin embargo, arrastra nodos diplicados con costes menores
                # quiza, lo recomando seria quitar nodos duplicados para mejorar memoria
                bandera_lista = False

                for nodo in lista_abierta: # se lo activa en cualqueira de los casos mencionados
                    if nodo == vecino and nodo.f <= vecino.f:
                        bandera_lista = True
                        break

                if not bandera_lista:
                    vecino.padre = nodo_actual
                    lista_abierta.append(vecino)
    return None, considerados
