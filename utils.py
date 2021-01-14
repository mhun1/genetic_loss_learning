import numpy as np
from deap import gp

def numpy_add(a,b):
    return a + b

def numpy_minus(a,b):
    return a - b

def numpy_mul(a,b):
    return a*b

def objective(a,b):
    return 2*(a+b) + 2*(a-b)

def fitness(pred,label):
    return np.linalg.norm(pred-label)

def plot_graph(expr):
    nodes, edges, labels = gp.graph(expr)

    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
