import matplotlib.pyplot as plt
from matriz import mapa
from algoritmos import a_estrella, search_bfs_dfs
from utils import punto_random
from metrics import Metrics, trace
from constanst import ANIMATION_DELAY

def render_maze(fig,mapa,camino, considerados):
    fig.imshow(mapa, cmap='binary')
    if considerados:
        for nodo in considerados:
            plt.pause(ANIMATION_DELAY)
            fig.plot(nodo.y, nodo.x, 'o',color='blue')
    if camino:
        for nodo in camino:
            plt.pause(ANIMATION_DELAY)
            fig.plot(nodo.y, nodo.x, 'o',color='red')

def main():
    fig_main, (f1,f2,f3) = plt.subplots(1,3, figsize = (12,4))
    punto_inicial = punto_random(mapa)
    meta = punto_random(mapa)

    #bfs

    bfsMetricas, (ultimo_nodo_bfs,considerados_bfs) = trace(
        search_bfs_dfs,
        mapa,
        'bfs',
        punto_inicial,
        meta
    )

    camino_bfs = ultimo_nodo_bfs.construir_camino()

    f1.set_title(f'BFS: {bfsMetricas}')
    render_maze(f1,mapa,camino_bfs,considerados_bfs)



    # estrella

    estrellaMetricas, (ultimo_nodo_estrella, estrella_considerados) = trace(
        a_estrella,
        mapa,
        punto_inicial,
        meta
    )
    f3.set_title(f'A*: {estrellaMetricas}')

    camino_estrella = ultimo_nodo_estrella.construir_camino()
    render_maze(f3,mapa,camino_estrella,estrella_considerados)
    plt.show()
    pass


if __name__ == '__main__':
    main()

