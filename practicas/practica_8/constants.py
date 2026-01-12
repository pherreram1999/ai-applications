from patterns import *

# lista de nuestros patrones
patrones_base = [
    patron_cruz,
    patron_cuadrado,
    patron_rombo,
    patron_estrella,
    patron_circulo,
    patron_triangulo
]

mapa_patrones = {
    "cruz": patron_cruz,
    "cuadrado": patron_cuadrado,
    "rombo": patron_rombo,
    "estrella": patron_estrella,
    "circulo": patron_circulo,
    "triangulo": patron_triangulo
}

NUMERO_COPIAS_POR_MUESTRA = 10

NUMERO_COPIAS_TEST_POR_MUESTRA = 3

Yd_cruz = [1, 0, 0, 0, 0, 0]
Yd_cuadrado = [0, 1, 0, 0, 0, 0]
Yd_rombo = [0, 0, 1, 0, 0, 0]
Yd_estrella = [0, 0, 0, 1, 0, 0]
Yd_circulo = [0, 0, 0, 0, 1, 0]
Yd_triangulo = [0, 0, 0, 0, 0, 1]

Yd_base = [
    Yd_cruz, # cruz
    Yd_cuadrado, # cuadrado
    Yd_rombo, # rombo
    Yd_estrella, # estrella
    Yd_circulo, # circulo
    Yd_triangulo, # triangulo
]



