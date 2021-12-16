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
    pass


def find_euler_cycle(graph):
    pass


def isBipartite(graph):
    pass


def isIsomorphic(graph1, graph2):
    pass


def graph_coloring(graph):
    pass


# can be used for searchig degree of vertices in graph
def dfs(graph: dict, node: int, path: list = []) -> list:
    """recurrent dfs algo for adjecency list presented graph

    Args:
        graph (dict): adjacency dict for graph
        vertex (int): starting vertex
        path (list, optional): path of dfs. Defaults to [].

    Returns:
        list: path of dfs from vertex
    """
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

    # import argparse

    # parser = argparse.ArgumentParser(description="swaping lines in file")
    # parser.add_argument("subline", help="line to swap")
    # parser.add_argument("swapline", help="line to swap with")
    # parser.add_argument("src", help="path to the file")
    # parser.add_argument("--inplace", help="change initial file", action="store_true")
    # args = parser.parse_args()
    start = time.perf_counter()
    # print(read_graph("graph_example.csv", repr_type="AdjMatrix",oriented=True))
    # print(read_graph("graph_example.csv", repr_type="AdjMatrix"))
    # print(read_graph("graph_example.csv","AdjList"))
    print(dfs(read_graph("graph_example.csv", "AdjDict"), 0))
    print(bfs(read_graph("graph_example.csv", "AdjDict"), 0))
    end = time.perf_counter()
    print(f"Time for execution function:{end-start}")
