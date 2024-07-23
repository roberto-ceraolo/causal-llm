# new_test
from openai import OpenAI
import dotenv
import numpy as np
import pandas as pd
from scipy import stats
import re
from typing import List, Dict, Any, Tuple
from sympy import sympify, symbols, Sum, Symbol, Basic
from sympy.stats import P, E
import json
import networkx as nx
import pandas as pd
import numpy as np
from dowhy import CausalModel
import hashlib
import json
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import os

# Constants
CACHE_FILE = "llm_cache.json"
MODEL = "gpt-4o-mini"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def hash_input(input_str: str) -> str:
    return hashlib.md5(input_str.encode()).hexdigest()

def get_cached_response(prompt: str, system_prompt: str = None) -> str:
    cache = load_cache()
    key = hash_input(prompt + (system_prompt or ""))
    return cache.get(key)

def cache_response(prompt: str, system_prompt: str, response: str):
    cache = load_cache()
    key = hash_input(prompt + (system_prompt or ""))
    cache[key] = response
    save_cache(cache)

def get_gpt_response(prompt: str, model: str = MODEL, system_prompt: str = None, response_format: Dict[str, Any] = None) -> str:
    cached_response = get_cached_response(prompt, system_prompt)
    if cached_response:
        print("Using cached response")
        return cached_response

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]

    caller = client.chat.completions
    response = caller.create(
        model=model,
        messages=messages,
        temperature=0.4,
        seed=42,
        response_format=response_format
    )
    response_content = response.choices[0].message.content
    
    cache_response(prompt, system_prompt, response_content)
    return response_content


def get_edges_from_pydot(graph):
    """
    Extracts edges from a pydot graph object and returns a dictionary representation.
    """
    edges = graph.get_edges()
    edges_list = []

    for edge in edges:
        source = edge.get_source()
        destination = edge.get_destination()
        edges_list.append({"source": source, "destination": destination})
    
    return edges_list


def get_pc_graph(df):
    """
    Get the PC graph from the data. Takes as input a pandas df, returns a string representation of the graph.
    """
    data = df.to_numpy()
    cg = pc(data)
    
    # Get column names
    column_names = df.columns.tolist()
    
    print(cg.G)

    # Get the graph as a pydot object
    pyd = GraphUtils.to_pydot(cg.G)

    # Get the edges
    edges = get_edges_from_pydot(pyd)

    # map the edges to the column names
    edges_mapped = []
    for edge in edges:
        source = column_names[int(edge["source"])]
        destination = column_names[int(edge["destination"])]
        edges_mapped.append({"source": source, "destination": destination})

    return edges_mapped
    

def generate_random_dataset(variables: set, outcome: str, n_samples: int = 1000):
    data = {}
    for var in variables:
        # Default to continuous (0, 1) for unknown variables
        data[var] = np.random.uniform(0, 1, n_samples)
    df = pd.DataFrame(data)
    
    # Generate Outcome based on other variables if it's in the variables
    if outcome in variables:
        predictors = [var for var in variables if var != outcome]
        coeffs = np.random.uniform(-1, 1, len(predictors))
        linear_combination = np.sum([coeff * df[var] for coeff, var in zip(coeffs, predictors)], axis=0)
        probability = 1 / (1 + np.exp(-linear_combination))
        df[outcome] = (np.random.random(n_samples) < probability).astype(int)
    
    return df

def generate_initial_dag(question: str) -> str:
    """
    Generates an initial DAG based on the given question and returns JSON.
    """
    
    prompt = f"""
    Given the following question about career choices:
    "{question}"
    Please identify the key variables involved in this decision, hypothesize causal relationships between these variables, including potential confounding factors. 
    Provide your response as a JSON object with the following structure:

    {{
        "variable_definitions": {{
            "ShortName1": "Full Definition 1",
            "ShortName2": "Full Definition 2",
            ...
        }},
        "causal_graph": [
            ["ShortName1", "ShortName2"],
            ["ShortName1", "ShortName3"],
            ["ShortName2", "Outcome"],
            ...
        ],
        "explanations": "Your explanation of the reasoning behind the causal graph structure and any assumptions made."
    }}

    Ensure that all variables used in the causal_graph are defined in the variable_definitions.
    """
    system_prompt = "You are a causal inference expert tasked with generating hypotheses about causal mechanisms in career decisions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    response_format = {"type": "json_object"}
    return get_gpt_response(prompt, system_prompt=system_prompt, response_format=response_format)

def parse_dag_output(output: str) -> Dict[str, Any]:
    """
    Parses the JSON output from the DAG generation function.
    
    Args:
    output (str): The JSON string output from the DAG generation function.
    
    Returns:
    dict: A dictionary containing the parsed JSON data.
    """
    # remove all "\n"
    output = output.replace("\n", "")


    try :
        parsed_data = json.loads(output)
        return parsed_data
    except json.JSONDecodeError:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, output, re.DOTALL)
        if match:
            parsed_data = match.group(1).strip()
            return json.loads(parsed_data)
        else:
            print("Error: GPT's response was not valid JSON. Raw response:")
            print(output)
            return None

def refine_dag(initial_dag: Dict[str, Any]) -> str:
    """
    Refines the initial DAG by considering additional factors and perspectives.
    
    Args:
    initial_dag (dict): A dictionary containing the initial DAG data.
    
    Returns:
    str: A JSON string containing the refined DAG output.
    """
    prompt = f"""
    Given the following initial causal graph for a career decision:

    {json.dumps(initial_dag, indent=2)}

    Please refine this graph by considering the following:
    1. Put yourself in the shoes of the decision-maker. What additional factors might they consider that aren't represented in the current graph?
    2. Consider potential long-term consequences that might not be immediately apparent.
    3. Think about external factors (e.g., economic conditions, technological advancements) that could influence the decision or its outcomes.
    4. Identify any feedback loops or bidirectional relationships that might exist.
    5. Consider any hidden variables that might be influencing multiple observed variables.

    Based on these considerations, provide your response as a JSON object with the following structure:

    {{
        "variable_definitions": {{
            "ShortName1": "Full Definition 1",
            "ShortName2": "Full Definition 2",
            ...
        }},
        "causal_graph": [
            ["ShortName1", "ShortName2"],
            ["ShortName1", "ShortName3"],
            ["ShortName2", "Outcome"],
            ...
        ],
        "treatment_variable": "TreatmentVarName",
        "outcome_variable": "OutcomeVarName",
        "explanations": "Your explanation of the reasoning behind the changes and additions to the graph, addressing the considerations mentioned above."
    }}

    Ensure that all variables used in the causal_graph are defined in the variable_definitions.
    """
    system_prompt = "You are a causal inference expert tasked with refining and improving causal graphs for career decisions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    response_format = {"type": "json_object"}
    return get_gpt_response(prompt, system_prompt=system_prompt, response_format=response_format)





def create_gml_graph(causal_graph: List[List[str]]) -> str:
    G = nx.DiGraph()
    for edge in causal_graph:
        G.add_edge(edge[0], edge[1])
    return nx.generate_gml(G)

def causal_effect_solver(question: str) -> Dict[str, Any]:
    # Generate and refine the DAG
    print("Generating initial DAG...")
    initial_dag = generate_initial_dag(question)
    parsed_initial_dag = parse_dag_output(initial_dag)
    var_definitions, initial_graph, initial_explanations = parsed_initial_dag['variable_definitions'], parsed_initial_dag['causal_graph'], parsed_initial_dag['explanations']
    print("Initial DAG generated. Variables: ", var_definitions)
    print("Causal Graph: ", initial_graph)
    print("Explanations: ", initial_explanations)

    # Refine the DAG based on additional factors
    print("Refining DAG...")
    refined_output = refine_dag(initial_dag)
    parsed_refined_dag = parse_dag_output(refined_output)
    refined_var_definitions, refined_graph, refined_explanations, treatment, outcome = parsed_refined_dag['variable_definitions'], parsed_refined_dag['causal_graph'], parsed_refined_dag['explanations'], parsed_refined_dag['treatment_variable'], parsed_refined_dag['outcome_variable']
    print("Refined Variable Definitions: ", refined_var_definitions)
    print("DAG refined. Refined DAG: ", refined_graph)
    print("Refined Explanations: ", refined_explanations)

    # Create GML graph for py-why
    gml_graph = create_gml_graph(refined_graph)

    # Generate random dataset based on the refined DAG
    variables = set(var for edge in refined_graph for var in edge)
    data = generate_random_dataset(variables, outcome)

    # Create CausalModel
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=gml_graph)

    # Identify causal effect
    print("Identifying causal effect...")
    identified_estimand = model.identify_effect()
    print("Identified estimand:", identified_estimand)

    # Estimate the causal effect
    print("Estimating causal effect...")
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.propensity_score_matching")
    print("Causal effect estimate:", estimate)

    # Refute the estimate
    print("Refuting estimate...")
    refute_results = model.refute_estimate(identified_estimand, estimate,
                                           method_name="random_common_cause")
    print("Refutation results:", refute_results)

    return {
        "initial_dag": initial_dag,
        "refined_dag": refined_graph,
        "identified_estimand": str(identified_estimand),
        "causal_estimate": str(estimate),
        "refutation_results": str(refute_results)
    }

dotenv.load_dotenv()
client = OpenAI()
question = "How does pursuing a graduate degree affect long-term career prospects?"
result = causal_effect_solver(question)
print(result)