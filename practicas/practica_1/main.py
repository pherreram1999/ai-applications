import sys
import networkx as nx # trabajar nodos y aristas
import matplotlib.pyplot as plt # plt

path = sys.argv[1]

print(path)

def graph_from_file(path):
    try:
        grafo = {}
        with open(path) as f:
            for line in f:
                l = line.strip()
                nodo, hijos = l.split(':')
                n = int(nodo)
                grafo[n] = []
                for hijo in hijos.split(','):
                    grafo[n].append(int(hijo))
        return grafo
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print("Ocurrio un error inesperado:", e)





def search_node(grafo, tipo, root_node, target_node):
    if tipo not in ['bfs', 'dfs']:
        raise Exception(f"tipo no valido: {tipo}, se espera bfs o dfs")

    graph_len = len(grafo)
    current_node = root_node
    stack = [root_node] # cola de lectura
    solve_it = False # indica si ya hemos encontrado la solucion

    readit = {}
    for k,_ in grafo.items():
        readit[k] = False

    #graficos
    G = nx.Graph()  # creamos el lienzo

    # almacenamos los colores de los nodos
    # en este caso cada posicion del arreglo representa un nodo
    node_colors = ['lightgreen'] * graph_len

    if root_node == target_node:
        solve_it = True
        node_colors[root_node] = 'blue'

    readit[root_node] = True # el nodo raiz ha sido leido

    positions = {
        current_node: (0, 0)  # nodo raiz
    }

    row  = 0
    col = 1

    while len(stack) > 0:
        if tipo == 'dfs':
            # se saca y quita el ultimo nodo
            current_node = stack.pop()
            row -= 1
            if len(stack) == 0: # salto de rama
                row = 1

        elif tipo == 'bfs':
            # se saca y qutia el primer nodo
            current_node = stack[0]
            stack = stack[1:]

        # preguntamos si el solucion
        if not solve_it and current_node == target_node:
            solve_it = True
            node_colors[current_node] = 'blue'

        if current_node != root_node:
            positions[current_node] = (col, row)


        for node in grafo[current_node]:
            if readit[node]: continue # nodo leido, saltamos
            stack.append(node)
            readit[node] = True #  indicamos que el nodo ha sido leido



    print(positions)





def main():
    grafo = graph_from_file(path)
    search_node(grafo, 'dfs', 1, 5)



if __name__ == "__main__":
    main()

