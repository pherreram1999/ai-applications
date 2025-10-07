import random
import numpy as np
from models import Nodo
from constanst import TRANSITABLE


def punto_random(mapa)-> Nodo:
    f,c = np.shape(mapa)
    while True:
        x = random.randint(0, f - 1)  # se resta 1 para evitar valores no indexados
        y = random.randint(0, c - 1)
        if mapa[x,y] == TRANSITABLE:
            return Nodo(x,y)
