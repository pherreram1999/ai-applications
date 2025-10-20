import numpy as np
import matplotlib.pyplot as plt
from laberinto import mapa


def render_map(fig,mapa):
    fig.set_title('A*')

    fig.imshow(mapa, cmap='binary')
    pass


if __name__ == "__main__":
    main_fig, (f1) = plt.subplots(1,1,figsize=(8,8))
    render_map(f1,mapa)
    plt.show()
    print(mapa)

