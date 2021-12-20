"""7. Graphs interaction"""
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import scipy.sparse as sp

# function can return oriented and non-oriented graphs in adjacency matrix, incidence matrix, adjacency list
def read_graph(
    path_to_file: str, repr_type: str = "AdjMatrix", oriented: bool = None
) -> np.ndarray:

    if oriented is None:
        oriented = False
    else:
        oriented = True
    graph = pd.read_csv(path_to_file, skiprows=0).to_numpy()
    graph = np.append(graph, np.full((np.shape(graph)[0], 1), [1]), axis=1)

    if repr_type == "AdjMatrix":
        adj_matrix = transform_to_adj_matrix(graph, oriented)
        return adj_matrix

    elif repr_type == "AdjDict":
        adj_dict = transform_to_adj_dict(graph, oriented)
        return adj_dict
    #     inc_matrix = sp.coo_matrix(
    #         (graph[:, 2], (graph[:, 0],graph[:,1])),
    #         shape=(graph.max()+1,np.shape(graph)),
    #         dtype=graph.dtype,
    # )
    #     return inc_matrix.todense()


def transform_to_adj_dict(graph: np.ndarray, oriented: bool) -> dict:
    """function for representation of graph edges in adjacency dict

    Args:
        graph (np.ndarray): lists of start - end points of edge and its weight
        oriented (bool): whether graph is oriented or not

    Returns:
        dict: adjacency dict for graph
    """
    adj_dict = {}
    if oriented:
        for edge in graph:
            adj_dict[edge[0]] = adj_dict.get(edge[0], []) + [edge[1]]
    else:
        for edge in graph:
            adj_dict[edge[0]] = adj_dict.get(edge[0], []) + [edge[1]]
            adj_dict[edge[1]] = adj_dict.get(edge[1], []) + [edge[0]]
    return adj_dict


def transform_to_adj_matrix(graph: np.ndarray, oriented: bool) -> np.ndarray:
    """Transforms edge list with weights into adjacency matrix

    Args:
        graph (np.ndarray): [out vertice,in vertice,weight] type of edge list
        oriented (bool): type of graph

    Returns:
        np.ndarray: adjacency matrix for graph
    """
    if oriented:
        in_deg = -1
    else:
        in_deg = 1
    shape = (graph.max() + 1, graph.max() + 1)
    adj_matrix = sp.coo_matrix(
        (graph[:, 2], (graph[:, 0], graph[:, 1])),
        shape=shape,
        dtype=graph.dtype,
    ) + sp.coo_matrix(
        (in_deg * graph[:, 2], (graph[:, 1], graph[:, 0])),
        shape=shape,
        dtype=graph.dtype,
    )
    return adj_matrix.todense()


def find_hamilton_cycle(graph):
    """Find the hamiltonian cycle in the graph
    Args:
        graph (dict): the graph

    Returns:
        path: the path of the hamiltonian cycle or False if there are no
        cycles
    """
    # Checking the amount of components, if the amount is over 2, than returns False
    if len(find_components(graph)) != 1:
        return False

    # The set of vertices that were visited, will be shown by True or False
    # Example: {False, False, True, False, True}
    visited = {}
    for elem in graph:
        visited[elem] = False
    start = list(graph.keys())[0]
    path = [start]
    visited[start] = True

    # The main function of the hamiltonian cycle. Uses backtracking
    def hamiltonian_cycle(path, vertix_index):
        # Checks if the len of the path is the len of the graph,
        # if so, returns the hamiltonian cycle
        if len(path) == len(graph):
            if start in graph[vertix_index]:
                path.append(start)
                return path
            else:
                pass

        # If the len of the path is less than len of the graph,
        # tries to create path through all the adjacency vertices
        for elem in graph[vertix_index]:
            if visited[elem] == False:
                visited[elem] = True
                path.append(elem)
                if hamiltonian_cycle(path, elem) != False:
                    return path
                visited[elem] = False
                path.remove(elem)
        return False

    return hamiltonian_cycle(path, start)

def find_euler_cycle(graph):
    pass


def isBipartite(graph):
    pass


def areIsomorphic(graph1:dict, graph2:dict)->bool:
    degree_invariant = set(check_degree(graph1).values()) == set(check_degree(graph2).values())
    components_invariant = len(find_components(graph1))== len(find_components(graph2))  
    return degree_invariant and components_invariant
        


def graph_coloring(graph):
    pass


def check_degree(graph:dict)->dict:
    vertices_degrees = {vertice: len(graph[vertice]) for vertice in graph}
    return vertices_degrees


def find_components(graph:dict)->list:
    components = []
    visited = []
    for vertice in graph:
        if vertice not in visited:
            component = set(dfs(graph,vertice))
            visited.extend(list(component))
            components.append(component)
    return components


# can be used for backtracking
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
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return path


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    # print(read_graph("graph_example.csv", repr_type="AdjMatrix",oriented=True))
    # print(read_graph("graph_example.csv", repr_type="AdjMatrix"))
    # print(read_graph("graph_example.csv","AdjList"))
    print(find_hamilton_cycle(read_graph("graph_example.csv", "AdjDict")))
    # print(dfs(read_graph("graph_example.csv", "AdjDict"), 2))
    # print(bfs(read_graph("graph_example.csv", "AdjDict"), 0))
    print(areIsomorphic(read_graph("graph_example.csv","AdjDict"),read_graph("graph_example.csv","AdjDict")))
    print(find_components(read_graph("graph_example.csv","AdjDict")))
    end = time.perf_counter()
    print(f"Time for execution function:{end-start}")
