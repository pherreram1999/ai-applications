import random
import numpy as np
from models import Nodo
from constanst import NO_TRANSITABLE


def punto_random(mapa)-> Nodo:
    f,c = np.shape(mapa)
    x = random.randint(0, f - 1) # se resta 1 para evitar valores no indexados
    y = random.randint(0, c - 1)
    if mapa[x, y] == NO_TRANSITABLE:
        return punto_random(mapa)
    return Nodo(x, y)