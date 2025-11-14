import matplotlib.pyplot as plt
from matriz import mapa
from algoritmos import a_estrella, search_bfs_dfs
from utils import punto_random
from metrics import trace
from constanst import ANIMATION_DELAY, START_COLOR, META_COLOR, COLOR_WAY, COLOR_MAP


def render_maze(fig,mapa,camino, considerados,punto_inicial, meta):
    fig.imshow(mapa, cmap='binary')

    # pintamos inicial
    fig.plot(punto_inicial.y,punto_inicial.x,'^',color=START_COLOR)
    fig.plot(meta.y,meta.x,'x',color=META_COLOR)

    # no se considera el inicio el final para pintarlos
    if considerados:
        for i in range(1,len(considerados) - 1):
            nodo = considerados[i]
            plt.pause(ANIMATION_DELAY)
            fig.plot(nodo.y, nodo.x, 'o',color=COLOR_MAP)

    if camino:
        for i in range(1, len(camino) - 1):
            nodo = camino[i]
            plt.pause(ANIMATION_DELAY)
            fig.plot(nodo.y, nodo.x, 'o', color=COLOR_WAY)



    pass

def main():
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

    # dfs

    dfsMetricas, (ultimo_nodo_dfs, considerados_dfs) = trace(
        search_bfs_dfs,
        mapa,
        'dfs',
        punto_inicial,
        meta
    )

    camino_dfs = ultimo_nodo_dfs.construir_camino()


    # estrella

    estrellaMetricas, (ultimo_nodo_estrella, estrella_considerados) = trace(
        a_estrella,
        mapa,
        punto_inicial,
        meta
    )

    camino_estrella = ultimo_nodo_estrella.construir_camino()

    # al final renderizamos

    fig_main, (f1,f2,f3) = plt.subplots(1,3, figsize = (18,7))
    fig_main.suptitle('Comparación entre BFS,DFS y A*\n\n')

    f1.set_title(f'BFS\n {bfsMetricas}')
    f2.set_title(f'DFS\n {dfsMetricas}')
    f3.set_title(f'A*\n {estrellaMetricas}')


    render_maze(f1, mapa, camino_bfs, considerados_bfs,punto_inicial, meta)
    render_maze(f2, mapa, camino_dfs, considerados_dfs,punto_inicial, meta)
    render_maze(f3,mapa,camino_estrella,estrella_considerados,punto_inicial, meta)
    plt.show()
    pass


if __name__ == '__main__':
    main()

