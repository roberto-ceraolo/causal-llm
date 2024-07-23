# causal_engine.py

from dowhy import CausalModel
import pandas as pd
from typing import Dict, Any
from llm import generate_initial_dag, parse_llm_dag_output, refine_dag, refine_dag_pc
from graph_utils import is_dag, create_gml_graph, save_causal_graph_png
from causal_utils import choose_causal_inference_method, estimate_causal_effect_gcm, get_pc_graph
from data_generation import generate_random_dataset
from config import SYNTHETIC_DATA
import networkx as nx

def causal_effect_solver(question: str, data: pd.DataFrame = None) -> Dict[str, str]:
    """
    Solve the causal effect problem given a question and a dataset.

    Args:
    question (str): The question about causal effects to be answered.
    data (pd.DataFrame): Optional. The dataset to be used for causal discovery and inference.

    Returns:
    dict: A dictionary containing the results of the causal effect analysis.
    """


    # Generate and refine the DAG
    print("Generating initial DAG...")
    initial_dag = generate_initial_dag(question)
    parsed_initial_dag = parse_llm_dag_output(initial_dag)
    var_definitions, initial_graph, initial_explanations = parsed_initial_dag['variable_definitions'], parsed_initial_dag['causal_graph'], parsed_initial_dag['explanations']
    print("Initial DAG generated. Variables: ", var_definitions)
    print("Causal Graph: ", initial_graph)
    print("Explanations: ", initial_explanations)

    

    # Refine the DAG based on additional factors (socratic reasoning)
    print("Refining DAG...")
    refined_output = refine_dag(question, parsed_initial_dag)
    parsed_refined_dag = parse_llm_dag_output(refined_output)
    refined_var_definitions, refined_graph, refined_explanations, treatment, outcome = parsed_refined_dag['variable_definitions'], parsed_refined_dag['causal_graph'], parsed_refined_dag['explanations'], parsed_refined_dag['treatment_variable'], parsed_refined_dag['outcome_variable']
    print("Refined Variable Definitions: ", refined_var_definitions)
    print("DAG refined. Refined DAG: ", refined_graph)
    print("Refined Explanations: ", refined_explanations)

    # Generate random dataset based on the refined DAG
    if data is None and SYNTHETIC_DATA:
        variables = set(var for edge in refined_graph for var in edge)
        data = generate_random_dataset(variables, outcome, treatment)

    if data is not None:
        pc_graph = get_pc_graph(data)
        print("PC Graph: ", pc_graph)
    else:
        pc_graph = None

    # Refine the graph on the basis of PC graph, if available
    if pc_graph:
        print("Refining DAG based on PC graph...")
        refined_output_pc = refine_dag_pc(question, refined_graph, pc_graph, treatment, outcome)
        parsed_refined_dag_pc = parse_llm_dag_output(refined_output_pc)
        refined_graph_pc, refined_explanations_pc  = parsed_refined_dag_pc['causal_graph'], parsed_refined_dag_pc['explanations']
        print("DAG refined based on PC graph. Refined DAG (PC): ", refined_graph_pc)
        print("Refined Explanations (PC): ", refined_explanations_pc)
        graph_is_dag, cycle = is_dag(refined_graph_pc)
        if graph_is_dag:
            refined_graph = refined_graph_pc
        else:
            print("Error: The refined graph based on PC graph contains a cycle. Using the graph refined only using reasoning.")
            print("Cycle: ", cycle)
            
    # Save the final causal graph as a PNG
    save_causal_graph_png(refined_graph, "final_causal_graph.png")

    # Create GML graph for py-why
    gml_graph = create_gml_graph(refined_graph)

    # Create CausalModel
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=gml_graph)

        # Convert GML graph to NetworkX DiGraph
    G = nx.parse_gml(gml_graph)

    # Identify causal effect
    print("Identifying causal effect...")
    identified_estimand = model.identify_effect()
    print("Identified estimand:", identified_estimand)

    # Choose causal inference method
    method, info = choose_causal_inference_method(data, G, treatment, outcome)
    print(f"Chosen estimation method: {method}")
    print(f"Reason: {info['reason']}")

    # Estimate the causal effect
    print("Estimating causal effect...")
    if method == "gcm":
        estimate_result = estimate_causal_effect_gcm(data, G, treatment, outcome)
    else:
        estimate = model.estimate_effect(identified_estimand, method_name=method)
        estimate_result = {
            "method": method,
            "causal_estimate": str(estimate)
        }

    print("Causal effect estimate:", estimate_result)


    return {
        "initial_dag": initial_dag,
        "refined_dag": refined_graph,
        "identified_estimand": str(identified_estimand),
        "estimation_method": method,
        "estimation_reason": info['reason'],
        "causal_estimate": estimate_result,
        "causal_graph_image": "final_causal_graph.png"
    }