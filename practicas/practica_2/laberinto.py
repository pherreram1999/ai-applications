import numpy
from mazelib import Maze
from mazelib.generate.Prims import Prims

# Crear laberinto 15x15
maze = Maze()
maze.generator = Prims(25, 25)  # Resultado será 15x15
maze.generate()

# Obtener array numpy
maze_array = maze.grid.astype(int)

print('[')
for row in maze_array:
    print('[', end='')
    for i,col in enumerate(row):
        print(col, end= '' if len(row) -1 == i  else ',')
    print('],', end='')
    print()
print(']')

