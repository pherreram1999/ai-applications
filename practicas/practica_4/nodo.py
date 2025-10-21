
class Nodo():
    def __init__(self,x, y, g = 0, f = 0, padre = None):
        self.x = x
        self.y = y
        # valores utilizados para la a estrella
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