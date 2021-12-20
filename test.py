import csv
def read_graph(path: str) -> list:
    """Return a graph, read from file,
    as a list of tuples. Each tuple
    represents an edge.

    Args:
        path (str): Path to csv file with graph

    Returns:
        list: Graph, read from file
    """

    graph = list()

    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for line in reader:
            graph.append((int(line[0]), int(line[1])))

    return graph

def create_adj_matrix(graph: list) -> dict:
    """Return adjacency matrix of a graph,
    given the list of it's edges.

    Args:
        graph (list): Graph as a list of edges

    Returns:
        dict: Adjacency matrix
    """

    adj_matrix = dict()

    for node1, node2 in graph:
        if node1 not in adj_matrix:
            adj_matrix[node1] = {node2}
        else:
            adj_matrix[node1].add(node2)

        if node2 not in adj_matrix:
            adj_matrix[node2] = {node1}
        else:
            adj_matrix[node2].add(node1)

    return adj_matrix
create_adj_matrix(read_graph("graph_100000_4998622_1.csv"))