import sys
import networkx as nx # trabajar nodos y aristas
import matplotlib.pyplot as plt # plt




COLOR_COMMON = 'lightgray'
COLOR_TARGET = 'lightgreen'
COLOR_WAY = 'lightblue'

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
    node_colors = [COLOR_COMMON] * len(grafo)
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
        node_colors[index_map[current_node]] = COLOR_TARGET

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
            node_colors[index_map[current_node]] = COLOR_TARGET
        elif not solve_it:
            node_colors[index_map[current_node]] = COLOR_WAY

        for node in grafo[current_node]:
            if readit[node]: continue # nodo leido, saltamos
            stack.append(node)
            readit[node] = True #  indicamos que el nodo ha sido leido
    # converitmos el mapa en una lista
    return node_colors


def calculate_positions(grafo: dict, root: int):
    """
    Calcula el las posiciones del nado para el grafo
    basandose en los niveles de grafo usando tenicas de BFS
    :param grafo:
    :param root:
    :return:
    """
    stack = [root]
    visited = {}
    for k,_ in grafo.items():
        visited[k] = False

    # por cada nivel del grafo, va ir aumentando
    level = 0
    pos = {}


    while len(stack) > 0:
        no_levels = len(stack)
        # dependiendo su longitud se centra o no el nodo
        # para que si empieza en 1 -1 = 0 / 2 = 0
        # 2 - 1 = 1 / 2 = .5 = -5
        x_start = -(no_levels - 1) / 2
        #print(f'x_start: {x_start} - no_level: {no_levels} - level: {level}')
        # recorremos los nodos relacionados
        for i in range (no_levels):
            node = stack.pop(0) # usamos bfs para ir por niveles
            y = -level
            x = x_start + i # va recoriendo de derecha a izquierda
            pos[node] = (x, y)
            #print(f'node: {node} pos: {pos[node]}')
            # indicamos que visitamos los nodos
            for child in grafo[node]:
                if visited[child]: continue
                visited[child] = True
                stack.append(child)
        level += 1

    # sobre escribimos dado que vuelve al primer nodo
    pos[root] = (0,0)
    return pos


def render_graph(graph: dict, target: int,colors: list, pos: dict):
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
    # con la posiciones del nodo calculadas, solo agarramos la del target,
    label_x,label_y = pos[target]
    # creamos un nuevo layout para dibjar la etiqueta
    nx.draw_networkx_labels(
        G,
        pos={target: (label_x + .2, label_y)},
        labels= {
            target: "solucion!"
        },
        font_size=10,
        font_weight='bold',
        font_color='black',
    )
    plt.show()


def main():
    path = sys.argv[1]
    target = 2
    root = 1

    grafo = graph_from_file(path)
    colors = search_node(grafo, 'dfs', root, target)
    pos = calculate_positions(grafo, root)
    render_graph(grafo, target,colors, pos)



if __name__ == "__main__":
    main()

