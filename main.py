"""7. Graphs interaction"""
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import scipy.sparse as sp

# function can return oriented and non-oriented graphs in adjacency matrix, incidence matrix, adjacency list
def read_graph(path_to_file: str, repr_type: str = "AdjMatrix", oriented: bool = None)->np.ndarray:
    if oriented == None:
        oriented = False
    else:
        oriented = True
    graph = pd.read_csv(path_to_file, skiprows=0).to_numpy()
    graph = np.append(graph, np.full((np.shape(graph)[0], 1), [1]), axis=1)

    if repr_type == "AdjMatrix":
        adj_matrix = transform_to_adj_matrix(graph,oriented)
        return adj_matrix

    elif repr_type == "IncMatrix":
        return "heeeelp"
    #     inc_matrix = sp.coo_matrix(
    #         (graph[:, 2], (graph[:, 0],graph[:,1])),
    #         shape=(graph.max()+1,np.shape(graph)),
    #         dtype=graph.dtype,
    # )
    #     return inc_matrix.todense()

def transform_to_adj_matrix(graph:np.ndarray,oriented:bool)->np.ndarray:
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
    print(read_graph("graph_example.csv", repr_type="AdjMatrix",oriented=True))
    print(read_graph("graph_example.csv", repr_type="AdjMatrix"))
    print(read_graph("graph_example.csv","IncMatrix"))
    end = time.perf_counter()
    print(f"Time for execution function:{end-start}")
