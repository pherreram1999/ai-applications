import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from constants import MAZE_COLORS, TRANSITABLE, AGUA, LODO, ANIMATION_DELAY, START_COLOR, META_COLOR, \
    CONSIDERADOS_COLOR, CAMINO_COLOR
from laberinto import mapa
from nodo import Nodo
from tracer import trace
from a_estrella import a_estrella


def render_map(fig,mapa, colors,camino,considerados,punto_inicial,meta):
    fig.set_title('A*')

    fig.imshow(mapa, cmap=colors)

    # pintamos inicial
    fig.plot(punto_inicial.y, punto_inicial.x, '^', color=START_COLOR)
    fig.plot(meta.y, meta.x, 'x', color=META_COLOR)

    # no se considera el inicio el final para pintarlos
    if considerados:
        for i in range(1, len(considerados) - 1):
            nodo = considerados[i]
            plt.pause(ANIMATION_DELAY)
            fig.plot(nodo.y, nodo.x, 'o', color=CONSIDERADOS_COLOR)

    if camino:
        for i in range(1, len(camino) - 1):
            nodo = camino[i]
            plt.pause(ANIMATION_DELAY)
            fig.plot(nodo.y, nodo.x, 'o', color=CAMINO_COLOR)
    pass


def get_random_point(mapa):
    """
    Regresa un punto random el mapa que sea transitable y no sea ni agua ni lodo
    :param mapa:
    :return:
    """
    f, c = np.shape(mapa)
    x = random.randint(0,c-1)
    y = random.randint(0,f-1)
    nodo = mapa[x,y]
    if nodo == TRANSITABLE and (nodo != AGUA and nodo != LODO):
        return Nodo(x,y)
    return get_random_point(mapa)

def run_a_estrella(fig,colors,nodo_inicial,meta):
    metricas, res = trace(
        a_estrella,
        mapa,
        nodo_inicial,
        meta
    )

    nodo_encontrado, considerados = res

    camino = None

    if nodo_encontrado:
        camino = nodo_encontrado.construir_camino()

    render_map(fig,mapa,colors,camino,considerados,nodo_inicial,meta)



if __name__ == "__main__":
    cmap_colors = mcolors.ListedColormap(MAZE_COLORS)

    main_fig, (f1) = plt.subplots(1,1,figsize=(8,8))

    inicio = get_random_point(mapa)
    meta = get_random_point(mapa)

    run_a_estrella(f1,cmap_colors,inicio,meta)

    plt.show()

