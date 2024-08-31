# causal_engine.py

from dowhy import CausalModel
import pandas as pd
from typing import Dict, Any
from llm import generate_initial_dag, interpret_causal_effect, parse_llm_dag_output, refine_dag, refine_dag_pc
from graph_utils import is_dag, create_gml_graph, save_causal_graph_png
from causal_utils import choose_causal_inference_method, estimate_causal_effect_gcm, get_pc_graph
from data_generation import generate_random_dataset
from config import DEBUG, SYNTHETIC_DATA
import networkx as nx
from kaggle_data_gathering import find_and_prepare_kaggle_dataset
from config import KAGGLE

import logging

# Set up logging
import datetime

log_filename = f'causal_solver_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def causal_effect_solver(question: str, data: pd.DataFrame = None) -> Dict[str, str]:
    """
    Solve the causal effect problem given a question and a dataset.

    Args:
    question (str): The question about causal effects to be answered.
    data (pd.DataFrame): Optional. The dataset to be used for causal discovery and inference.

    Returns:
    dict: A dictionary containing the results of the causal effect analysis.
    """
    logging.info(f"Starting causal effect solver for question: {question}")

    # Generate and refine the DAG
    logging.info("Generating initial DAG...")
    initial_dag = generate_initial_dag(question)
    parsed_initial_dag = parse_llm_dag_output(initial_dag)
    var_definitions, initial_graph, initial_explanations = parsed_initial_dag['variable_definitions'], parsed_initial_dag['causal_graph'], parsed_initial_dag['explanations']
    logging.info(f"Initial DAG generated. Variables: {var_definitions}")
    logging.info(f"Causal Graph: {initial_graph}")
    logging.info(f"Explanations: {initial_explanations}")

    # Refine the DAG based on additional factors (socratic reasoning)
    logging.info("Refining DAG...")
    refined_output = refine_dag(question, parsed_initial_dag)
    parsed_refined_dag = parse_llm_dag_output(refined_output)
    refined_var_definitions, refined_graph, refined_explanations, treatment, outcome = parsed_refined_dag['variable_definitions'], parsed_refined_dag['causal_graph'], parsed_refined_dag['explanations'], parsed_refined_dag['treatment_variable'], parsed_refined_dag['outcome_variable']
    logging.info(f"Refined Variable Definitions: {refined_var_definitions}")
    logging.info(f"DAG refined. Refined DAG: {refined_graph}")
    logging.info(f"Refined Explanations: {refined_explanations}")

    # Generate random dataset based on the refined DAG
    if data is None:
        dag_variables = list(set([var for edge in refined_graph for var in edge]))
        df = None
        if KAGGLE:
            # Load embeddings
            index, dataset_info = load_embeddings()
            # Use the loaded embeddings for finding datasets
            df = find_and_prepare_kaggle_dataset(treatment, outcome, dag_variables, index, dataset_info)
        if df is not None:
            data = df
            logging.info("Using Kaggle dataset for causal inference")
        elif SYNTHETIC_DATA:
            data = generate_random_dataset(dag_variables, outcome, treatment)
            logging.info("Generated synthetic dataset")
        else:
            logging.error("No dataset found on Kaggle and no synthetic data was generated.")
            data = None

    if data is not None:
        pc_graph = get_pc_graph(data)
        logging.info(f"PC Graph: {pc_graph}")
    else:
        pc_graph = None
        
    if DEBUG:
        pc_graph = None

    # Refine the graph on the basis of PC graph, if available
    if pc_graph:
        logging.info("Refining DAG based on PC graph...")
        refined_output_pc = refine_dag_pc(question, refined_graph, pc_graph, treatment, outcome)
        parsed_refined_dag_pc = parse_llm_dag_output(refined_output_pc)
        refined_graph_pc, refined_explanations_pc  = parsed_refined_dag_pc['causal_graph'], parsed_refined_dag_pc['explanations']
        logging.info(f"DAG refined based on PC graph. Refined DAG (PC): {refined_graph_pc}")
        logging.info(f"Refined Explanations (PC): {refined_explanations_pc}")
        graph_is_dag, cycle = is_dag(refined_graph_pc)
        if graph_is_dag:
            refined_graph = refined_graph_pc
        else:
            logging.warning("Error: The refined graph based on PC graph contains a cycle. Using the graph refined only using reasoning.")
            logging.warning(f"Cycle: {cycle}")

    # Save the final causal graph as a PNG
    save_causal_graph_png(refined_graph, "final_causal_graph.png")
    logging.info("Saved final causal graph as PNG")

    if data is not None:
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
        logging.info("Identifying causal effect...")
        identified_estimand = model.identify_effect()
        logging.info(f"Identified estimand: {identified_estimand}")

        # Choose causal inference method
        method, info = choose_causal_inference_method(data, G, treatment, outcome, question)
        logging.info(f"Chosen estimation method: {method}")
        logging.info(f"Reason: {info['reason']}")

        # Estimate the causal effect
        logging.info("Estimating causal effect...")
        if method == "gcm":
            estimate_result = estimate_causal_effect_gcm(data, G, treatment, outcome)
        else:
            estimate = model.estimate_effect(identified_estimand, method_name=method)
            estimate_result = {
                "method": method,
                "causal_estimate": str(estimate)
            }

        logging.info(f"Causal effect estimate: {estimate_result}")

        # Add interpretation step
        interpretation = interpret_causal_effect(question, treatment, outcome, estimate_result)
        logging.info(f"Interpretation: {interpretation}")
    else:
        interpretation = "No data was provided, so no causal effect could be estimated."

    result = {
        "initial_dag": parsed_initial_dag,
        "refined_dag": parsed_refined_dag,
        "identified_estimand": str(identified_estimand) if data is not None else None,
        "estimation_method": method if data is not None else None,
        "estimation_reason": info['reason'] if data is not None else None,
        "causal_estimate": estimate_result if data is not None else None,
        "interpretation": interpretation if data is not None else None,
        "causal_graph_image": "final_causal_graph.png",
        "log_filename": log_filename
    }
    
    logging.info("Causal effect solver completed")
    return result