# causal_discovery.py

from graph_utils import get_edges_from_pydot
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from typing import List
import pandas as pd

def get_pc_graph(df: pd.DataFrame) -> List[List[str]]:
    """
    Get the PC graph from the data. Takes as input a pandas df, returns a string representation of the graph. 

    Args:
        df: pandas DataFrame

    Returns:
        List of edges in the graph.
    """
    data = df.to_numpy()
    cg = pc(data)

    column_names = df.columns.tolist()
    pyd = GraphUtils.to_pydot(cg.G)
    edges = get_edges_from_pydot(pyd)
    edges_mapped = []

    # converts the nodes to the column names
    for edge in edges:
        source = column_names[int(edge["source"])]
        destination = column_names[int(edge["destination"])]
        edges_mapped.append([source, destination])

    return edges_mapped
