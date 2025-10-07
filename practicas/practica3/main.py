import matplotlib.pyplot as plt
from matriz import mapa
from algoritmos import a_estrella
from utils import punto_random
from metrics import Metrics, trace
from constanst import ANIMATION_DELAY

def render_maze(fig,mapa,camino, considerados):
    fig.imshow(mapa, cmap='binary')
    if considerados:
        for nodo in considerados:
            fig.plot(nodo.y, nodo.x, 'o',color='blue')
    if camino:
        for nodo in camino:
            fig.plot(nodo.y, nodo.x, 'o',color='red')

def main():
    fig_main, (f1,f2,f3) = plt.subplots(1,3, figsize = (12,4))
    punto_inicial = punto_random(mapa)
    meta = punto_random(mapa)

    # estrella
    ultimo_nodo_estrella, estrellaConsiderados = a_estrella(mapa,punto_inicial,meta)
    camino_estrella = ultimo_nodo_estrella.construir_camino()
    render_maze(f1,mapa,camino_estrella,estrellaConsiderados)
    plt.show()
    pass


if __name__ == '__main__':
    main()

