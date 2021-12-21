"""Module that finds colorization of given graph
    """

import main


def click_4_check(graph: dict) -> tuple:
    for first_vert in graph.keys():
        for second_vert in graph[first_vert]:
            for third_vert in graph[second_vert]:
                if third_vert != first_vert and first_vert in graph[third_vert]:
                    for fourth_vert in graph[third_vert]:
                        if (
                            fourth_vert != second_vert
                            and fourth_vert != first_vert
                            and first_vert in graph[fourth_vert]
                            and second_vert in graph[fourth_vert]
                        ):
                            return (
                                False,
                                (first_vert, second_vert, third_vert, fourth_vert),
                            )

    return (True, ())


def colorize(graph: list, vert: int, spread: list):
    colors = 3

    for color in range(colors):
        if safe_vert(graph, vert, color, spread):
            spread[vert] = color
            if vert + 1 < len(spread):
                spread = colorize(graph, vert + 1, spread)
            else:
                return spread
    
    return spread


def safe_vert(graph: list, vert: int, color: int, spread: list) -> bool:
    for i in range(len(graph)):
        if graph[vert][i] != 0 and spread[i] == color:
            return False

    return True


def main_colorize(file_path: str,orient:bool = None):
    # orient = False if orient is None else True
    if not click_4_check(main.read_graph(file_path, repr_type='AdjDict', oriented=orient))[0]:
        print('Coloring is not possible, 4-click exists in graph')
        return False

    G = main.read_graph(file_path, repr_type="AdjMatrix",oriented=orient).getA()
    spread = [None for i in range(len(G))]

    return colorize(G, 0, spread)


if __name__ == "__main__":
    print(main_colorize('graph_example.csv'))
    # print(main.read_graph('graph_example.csv').keys())
    # print(click_4_check(main.read_graph('graph_example.csv', repr_type='AdjDict')))
