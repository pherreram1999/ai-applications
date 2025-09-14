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

    current_node = root_node
    stack = [root_node] # cola de lectura
    solve_it = False # indica si ya hemos encontrado la solucion
    readit = {}
    node_colors = ['lightgray'] * len(grafo)
    for k,_ in grafo.items():
        readit[k] = False

    index_map = {}

    index = 0
    for k,_ in grafo.items():
        index_map[k] = index
        index += 1

    # almacenamos los colores de los nodos
    # en este caso cada posicion del arreglo representa un nodo

    if root_node == target_node:
        solve_it = True
        node_colors[root_node] = 'blue'

    readit[root_node] = True # el nodo raiz ha sido leido


    while len(stack) > 0:
        if tipo == 'dfs':
            # se saca y quita el ultimo nodo
            current_node = stack.pop()
        elif tipo == 'bfs':
            # se saca y qutia el primer nodo
            current_node = stack.pop(0)

        print(f'node: {current_node}')

        # preguntamos si el solucion
        if not solve_it and current_node == target_node:
            solve_it = True
            node_colors[index_map[current_node]] = 'lightblue'
        elif not solve_it:
            node_colors[index_map[current_node]] = 'lightgreen'

        for node in grafo[current_node]:
            if readit[node]: continue # nodo leido, saltamos
            stack.append(node)
            readit[node] = True #  indicamos que el nodo ha sido leido
    # converitmos el mapa en una lista
    print(node_colors)
    return node_colors


def calculate_positions(grafo: dict, root: int):
    stack = [root]
    visited = {}
    for k,_ in grafo.items():
        visited[k] = False
    level = 0

    pos = {}


    while len(stack) > 0:
        no_levels = len(stack)
        # dependiendo su longitud se centra o no el nodo
        x_start = -(no_levels - 1) / 2
        # recorremos los nodos relacionados
        for i in range (no_levels):
            node = stack.pop(0) # usamos bfs para ir por niveles
            y = -level
            x = x_start + i
            pos[node] = (x, y)
            # indicamos que visitamos los nodos
            for child in grafo[node]:
                if visited[child]: continue
                visited[child] = True
                stack.append(child)
        level += 1

    pos[root] = (0,0)
    return pos


def render_graph(graph: dict, colors: list, pos: dict):
    G = nx.Graph()
    for node, children in graph.items():
        for child in children:
            G.add_edge(node, child)

    nx.spring_layout(G)
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        font_weight='bold',
        node_color=colors,
        node_size=200,
        font_size=10,
    )
    plt.show()


def main():
    grafo = graph_from_file(path)
    colors = search_node(grafo, 'dfs', 1, 6)
    pos = calculate_positions(grafo, 1)
    render_graph(grafo, colors, pos)



if __name__ == "__main__":
    main()

