"""7. Graphs interaction"""
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import scipy.sparse as sp
from itertools import permutations
from collections import Counter
from csv import reader

# function can return oriented and non-oriented graphs in adjacency matrix, incidence matrix, adjacency list
def read_graph(
    path_to_file: str, repr_type: str = "AdjDict", oriented: bool = None
) -> dict:

    oriented = False if oriented is None else True

    if repr_type == "AdjMatrix":
        graph = pd.read_csv(path_to_file, skiprows=0, delimiter=" ").to_numpy()
        graph = np.append(graph, np.full((np.shape(graph)[0], 1), [1]), axis=1)
        return transform_to_adj_matrix(graph, oriented)

    elif repr_type == "AdjDict":
        graph = list()

        with open(path_to_file, "r") as file:
            read_data = reader(file, delimiter=" ")
            for line in read_data:
                graph.append((int(line[0]), int(line[1])))
        return transform_to_adj_dict(graph, oriented)


def transform_to_adj_dict(graph: list, oriented: bool) -> dict:
    """function for representation of graph edges in adjacency dict

    Args:
        graph (np.ndarray): lists of start - end points of edge and its weight
        oriented (bool): whether graph is oriented or not

    Returns:
        dict: adjacency dict for graph
    """
    adj_dict = dict()
    if oriented:
        for edge in graph:
            if edge[0] not in adj_dict:
                adj_dict[edge[0]] = {edge[1]}
            else:
                adj_dict[edge[0]].add(edge[1])

    else:
        for edge in graph:
            if edge[0] not in adj_dict:
                adj_dict[edge[0]] = {edge[1]}
            else:
                adj_dict[edge[0]].add(edge[1])

            if edge[1] not in adj_dict:
                adj_dict[edge[1]] = {edge[0]}
            else:
                adj_dict[edge[1]].add(edge[0])

    return adj_dict


def transform_to_adj_matrix(graph: np.ndarray, oriented: bool) -> np.ndarray:
    """Transforms edge list with weights into adjacency matrix

    Args:
        graph (np.ndarray): [out vertice,in vertice,weight] type of edge list
        oriented (bool): type of graph

    Returns:
        np.ndarray: adjacency matrix for graph
    """
    in_deg = -1 if oriented else 1
    shape = (graph.max(), graph.max())

    adj_matrix = sp.coo_matrix(
        (graph[:, 2], (graph[:, 0]-1, graph[:, 1]-1)),
        shape=shape,
        dtype=graph.dtype,
    ) + sp.coo_matrix(
        (in_deg * graph[:, 2], (graph[:, 1]-1, graph[:, 0]-1)),
        shape=shape,
        dtype=graph.dtype,
    )
    return adj_matrix.todense()


def find_hamilton_cycle(graph: dict) -> list:
    """Find the hamiltonian cycle in the graph
    Args:
        graph (dict): the graph

    Returns:
        path: the path of the hamiltonian cycle or False if there are no
        cycles
    """
    if len(find_components(graph)) != 1:
        return False
    visited = {}
    for elem in graph:
        visited[elem] = False
    start = list(graph.keys())[0]
    path = [start]
    visited[start] = True

    def hamiltonian_cycle(path: list, v: int) -> list:
        if len(path) == len(graph):
            if start in graph[v]:
                path.append(start)
                return path
            else:
                pass
        for elem in graph[v]:
            if visited[elem] == False:
                visited[elem] = True
                path.append(elem)
                if hamiltonian_cycle(path, elem) != False:
                    return path
                visited[elem] = False
                path.remove(elem)
        return False

    return hamiltonian_cycle(path, start)


def find_euler_cycle(graph: dict) -> list:
    """Find the euler cycle in the graph

    Args:
        graph (dict): the graph

    Returns:
        path: the path of the euler cycle or False if there are no
        cycles

    """
    def euler_cycle(graph: dict) -> list:

        path = []
        queue = [list(graph.keys())[0]]
        while queue:
            vertex = queue[-1]
            if graph[vertex]:
                adj_vertex = graph[vertex][0]
                queue.append(adj_vertex)
                graph.get(vertex).remove(adj_vertex)
                graph.get(adj_vertex).remove(vertex)
            else:
                path.append(queue.pop())
        return path

    if len(find_components(graph)) != 1:
        return False

    for i in check_degree(graph).values():
        if i % 2 != 0:
            return False
    return euler_cycle(graph)


def isBipartite(graph):
    pass


def areIsomorphic(graph1: dict, graph2: dict) -> bool:
    """checks whether two graphs are isomorphic

    Args:
        graph1 (dict): adjacency dict of graph
        graph2 (dict): adjacency dict of graph

    Returns:
        bool: whether two graphs are isomorphic
    """

    def degree_invariant(graph1: dict, graph2: dict) -> bool:
        """checks and returns whether two graphs have similar degrees of vertices"""
        degree_graph1 = list(check_degree(graph1).values())
        degree_graph2 = list(check_degree(graph2).values())
        return Counter(degree_graph1) == Counter(degree_graph2)

    def components_invariant(graph1: dict, graph2: dict) -> bool:
        """checks and returns whether two graphs have same number of components
        and length of that components"""
        components_graph1 = list(map(lambda x: len(x), find_components(graph1)))
        components_graph2 = list(map(lambda x: len(x), find_components(graph2)))
        return Counter(components_graph1) == Counter(components_graph2)

    def hamilton_invariant(graph1: dict, graph2: dict) -> bool:
        """checks whether two graphs have or don't have hamiltonian cycles"""
        hamilton_graph1 = bool(find_hamilton_cycle(graph1))
        hamilton_graph2 = bool(find_hamilton_cycle(graph2))
        return hamilton_graph1 == hamilton_graph2

    def euler_invariant(graph1: dict, graph2: dict) -> bool:
        euler_graph1 = bool(find_euler_cycle(graph1))
        euler_graph2 = bool(find_euler_cycle(graph2))
        return euler_graph1 == euler_graph2

    if not (
        degree_invariant(graph1, graph2)
        and components_invariant(graph1, graph2)
        and hamilton_invariant(graph1, graph2)
        and euler_invariant(graph1, graph2)
    ):
        return False
    else:
        vertices_permutations = permutations(range(1, len(graph1) + 1))
        graph1 = {node:list(adj_points) for node,adj_points in graph1.items()}
        graph2 = {node:list(adj_points) for node,adj_points in graph2.items()}
        for permutation in vertices_permutations:
            new_dict = {}
            for i, vert in enumerate(permutation):
                points = graph1[vert].copy()
                for j, point in enumerate(points):
                    points[j] = permutation.index(point) + 1
                new_dict[i + 1] = points
            if new_dict == graph2:
                return True
        return False


def graph_coloring(graph):
    pass


def check_degree(graph: dict) -> dict:
    vertices_degrees = {vertice: len(graph[vertice]) for vertice in graph}
    return vertices_degrees


def find_components(graph: dict) -> list:
    components = []
    visited = []
    for vertice in graph:
        if vertice not in visited:
            component = set(dfs(graph, vertice))
            visited.extend(list(component))
            components.append(component)
    return components


def dfs(graph: dict, node: int, path: list = None) -> list:
    """recurrent dfs algo for adjecency list presented graph

    Args:
        graph (dict): adjacency dict for graph
        vertex (int): starting vertex
        path (list, optional): path of dfs. Defaults to [].

    Returns:
        list: path of dfs from vertex
    """
    if path == None:
        path = []
    path.append(node)
    if node in graph:
        for neighbour in graph[node]:
            if neighbour not in path:
                path = dfs(graph, neighbour, path)
    return path


# can be used for searchig degree of vertices in graph
# also useful in searching components and cycles and for backtracking in general
def bfs(graph: dict, node: int) -> list:
    """bfs algo for adjacency dict presented graph

    Args:
        graph (dict): adjacency dict for graph
        node (int): starting node

    Returns:
        list: path of bfs
    """
    visited = []
    queue = []
    path = []
    visited.append(node)
    queue.append(node)
    while queue:
        s = queue.pop(0)
        path.append(s)
        if s in graph:
            for neighbour in graph[s]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
    return path


if __name__ == "__main__":
    import time

    start = time.perf_counter()

    # print(read_graph("graph_example.csv", repr_type="AdjDict"))
    # print(read_graph("graph_100000_4998622_1.csv", repr_type="AdjDict"))
    # read_graph("graph_100000_4998622_1.csv", "AdjDict")
    print(read_graph("graph_example.csv", repr_type="AdjMatrix"))
    print(bfs(read_graph("graph_example.csv", "AdjDict", oriented=True), 1))
    # # print(bfs(read_graph("graph_example.csv", "AdjDict"), 0))
    print(
        areIsomorphic(
            read_graph("graph_example.csv", "AdjDict"), read_graph("iso.csv", "AdjDict")
        )
    )
    print(find_components(read_graph("graph_example.csv","AdjDict",oriented=True)))
    print(find_hamilton_cycle(read_graph("graph_example.csv", "AdjDict")))
    print(find_euler_cycle(read_graph("graph_example.csv", "AdjDict")))
    end = time.perf_counter()
    print(f"Time for execution function:{end-start}")
