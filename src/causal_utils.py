# causal_discovery.py

from graph_utils import get_edges_from_pydot
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from typing import List
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple
import dowhy.gcm as gcm


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

def choose_causal_inference_method(data: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str) -> Tuple[str, Dict[str, Any]]:
    """
    Choose the most appropriate causal inference method based on data and graph characteristics.

    Args:
    data (pd.DataFrame): The dataset containing treatment, outcome, and covariates.
    graph (nx.DiGraph): The causal graph as a NetworkX DiGraph object.
    treatment (str): The name of the treatment variable.
    outcome (str): The name of the outcome variable.

    Returns:
    Tuple[str, Dict[str, Any]]: A tuple containing the chosen method and a dictionary of additional information.
    """
    # Extract relevant information
    n_samples, n_features = data.shape
    n_edges = len(graph.edges())
    treatment_values = data[treatment].unique()
    
    # Check data characteristics
    is_binary_treatment = len(treatment_values) == 2
    is_multi_valued_treatment = len(treatment_values) > 2
    is_continuous_treatment = data[treatment].dtype in ['float64', 'float32']
    is_continuous_outcome = data[outcome].dtype in ['float64', 'float32']
    
    # Check graph characteristics
    has_unobserved_confounders = any(node.startswith('U') for node in graph.nodes())
    has_instrumental_variables = any(len(list(graph.successors(node))) == 1 and list(graph.successors(node))[0] == treatment for node in graph.predecessors(treatment))
    
    # Check for complex relationships
    has_nonlinear_relationships = check_for_nonlinearity(data, treatment, outcome)
    has_interactions = check_for_interactions(data, treatment, outcome)
    
    # Decision logic
    if has_unobserved_confounders:
        if has_instrumental_variables:
            return "iv.instrumental_variable", {"reason": "Unobserved confounders present, but instrumental variables available"}
        else:
            return "gcm", {"reason": "Unobserved confounders present, GCM can handle this scenario"}
    
    if is_multi_valued_treatment or is_continuous_treatment:
        return "gcm", {"reason": "GCM can handle multi-valued or continuous treatments"}
    
    if has_nonlinear_relationships or has_interactions:
        return "gcm", {"reason": "GCM can capture complex relationships including nonlinearities and interactions"}
    
    if n_edges > 5 and n_features > 10:
        return "gcm", {"reason": "Complex graph structure and high-dimensional data"}
    
    if is_binary_treatment and is_continuous_outcome:
        if n_samples > 10 * n_features:
            return "backdoor.propensity_score_matching", {"reason": "Sufficient sample size for propensity score methods"}
        else:
            return "backdoor.linear_regression", {"reason": "Limited sample size, using simpler method"}
    
    # Default to GCM for other scenarios
    return "gcm", {"reason": "Default choice for unspecified scenarios"}

def check_for_nonlinearity(data: pd.DataFrame, treatment: str, outcome: str) -> bool:
    """
    Check for potential nonlinear relationships between treatment and outcome.
    """
    from scipy import stats
    
    # Perform a simple check using Spearman correlation
    spearman_corr, _ = stats.spearmanr(data[treatment], data[outcome])
    pearson_corr, _ = stats.pearsonr(data[treatment], data[outcome])
    
    # If Spearman correlation is significantly different from Pearson, it might indicate nonlinearity
    return abs(spearman_corr - pearson_corr) > 0.1

def check_for_interactions(data: pd.DataFrame, treatment: str, outcome: str) -> bool:
    """
    Check for potential interactions between treatment and other variables.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import partial_dependence
    
    # Prepare the data
    X = data.drop(columns=[outcome])
    y = data[outcome]
    
    # Fit a random forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Check partial dependence for treatment variable
    pd_results = partial_dependence(rf, X, features=[treatment])
    
    # If partial dependence is not monotonic, it might indicate interactions
    return not is_monotonic(pd_results['average'][0])

def is_monotonic(arr: np.ndarray) -> bool:
    """
    Check if an array is monotonically increasing or decreasing.
    """
    return (all(arr[i] <= arr[i+1] for i in range(len(arr)-1)) or
            all(arr[i] >= arr[i+1] for i in range(len(arr)-1)))

def estimate_causal_effect_gcm(data: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str) -> Dict[str, Any]:
    """
    Estimate the average causal effect using the Generalized Causal Model (GCM) approach.

    Args:
    data (pd.DataFrame): The dataset containing treatment, outcome, and covariates.
    graph (nx.DiGraph): The causal graph as a NetworkX DiGraph object.
    treatment (str): The name of the treatment variable.
    outcome (str): The name of the outcome variable.

    Returns:
    Dict[str, Any]: A dictionary containing the estimated causal effect and additional information.
    """
    # Create a ProbabilisticCausalModel
    causal_model = gcm.ProbabilisticCausalModel(graph)

    # Assign and fit causal mechanisms
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    gcm.fit(causal_model, data)

    # Estimate the average causal effect
    ace = gcm.average_causal_effect(
        causal_model,
        outcome,
        interventions_alternative={treatment: lambda x: 1},
        interventions_reference={treatment: lambda x: 0},
        num_samples_to_draw=1000
    )

    return {
        "method": "gcm",
        "average_causal_effect": ace,
    }
