# graph_utils.py

import networkx as nx
from typing import List, Dict
import pydot
import matplotlib.pyplot as plt

def is_dag(graph: List[List[str]]) -> bool:
    """
    Check if the graph contains a cycle.

    Args:
        graph: List of edges in the graph.

    Returns:
        True if the graph is a DAG, False otherwise.
        The cycle if the graph is not a DAG.
    """
    G = nx.DiGraph()
    for edge in graph:
        G.add_edge(edge[0], edge[1])
    is_dag = nx.is_directed_acyclic_graph(G)
    if is_dag:
        cycle = None
    else:
        cycle = nx.find_cycle(G)
    return is_dag, cycle

def get_edges_from_pydot(graph: pydot.Dot) -> List[Dict[str, str]]:
    """
    Extracts edges from a pydot graph object and returns a dictionary representation.

    Args:
        graph: pydot graph object.

    Returns:
        List of edges in the graph.

    """
    edges = graph.get_edges()
    edges_list = []

    for edge in edges:
        source = edge.get_source()
        destination = edge.get_destination()
        edges_list.append({"source": source, "destination": destination})
    
    return edges_list

def create_gml_graph(causal_graph: List[List[str]]) -> str:
    """
    Creates a GML representation of the causal graph.

    Args:
        causal_graph: List of edges in the causal graph.

    Returns:
        GML representation of the causal graph.
    """
    G = nx.DiGraph()
    for edge in causal_graph:
        G.add_edge(edge[0], edge[1])
    return "".join(nx.generate_gml(G))

def save_causal_graph_png(causal_graph: List[List[str]], filename: str = "causal_graph.png") -> None:
    """
    Create and save a PNG image of the causal graph.

    Args:
    causal_graph (List[List[str]]): A list of edges in the causal graph.
    filename (str): The filename to save the PNG image (default: "causal_graph.png").

    Returns:
    None
    """
    G = nx.DiGraph()
    for edge in causal_graph:
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', 
            arrows=True, edge_color='gray')

    # Add edge labels
    edge_labels = {(edge[0], edge[1]): "" for edge in causal_graph}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Causal Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Causal graph saved as {filename}")