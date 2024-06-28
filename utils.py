# utils
import re
import dotenv
from openai import OpenAI
import graphviz
import pandas as pd
from scipy import stats
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from sklearn.linear_model import LinearRegression
import json
from sklearn.linear_model import LogisticRegression


default_q = "Should I do my PhD with prof. Schölkopf or with prof. Tenenbaum, if afterwards, I want to get hired as a professor at ETH Zurich?"

cached_answer = """
1. Key Variables Involved in the Decision
Advisor (Prof. Schölkopf or Prof. Tenenbaum)
Research Quality
Publication Record
Networking Opportunities
Research Fit with Harvard's Interests
Reputation of the Advisor
External Recommendations
Harvard Hiring Criteria
Job Offer at Harvard (Outcome)
2. Hypothesized Causal Relationships Between These Variables
Advisor (Prof. Schölkopf or Prof. Tenenbaum)
Directly impacts Research Quality, Publication Record, Networking Opportunities, Reputation of the Advisor, and External Recommendations.
Research Quality
Directly impacts Publication Record, Reputation of the Advisor, and Research Fit with Harvard's Interests.
Publication Record
Directly impacts External Recommendations and Harvard Hiring Criteria.
Networking Opportunities
Directly impacts External Recommendations and Reputation of the Advisor.
Research Fit with Harvard's Interests
Directly impacts Harvard Hiring Criteria.
Reputation of the Advisor
Directly impacts External Recommendations and Harvard Hiring Criteria.
External Recommendations
Directly impacts Harvard Hiring Criteria.
Harvard Hiring Criteria
Directly impacts Job Offer at Harvard (Outcome).
3. Potential Confounding Factors
Field of Study: Certain fields may have higher demand at Harvard regardless of the advisor.
Personal Connections: Previous connections or networking outside of the advisor's network.
Institutional Bias: Harvard's internal preferences for candidates from specific universities or advisors.
Geographical Location: Proximity to Harvard and related social/professional circles.
Previous Work Experience: Prior work experience may independently influence hiring decisions.
4. Suggested Causal Graph Structure
Advisor -> Research Quality
Advisor -> Publication Record
Advisor -> Networking Opportunities
Advisor -> Reputation of the Advisor
Advisor -> External Recommendations
Research Quality -> Publication Record
Research Quality -> Research Fit with Harvard's Interests
Publication Record -> External Recommendations
Publication Record -> Harvard Hiring Criteria
Networking Opportunities -> External Recommendations
Networking Opportunities -> Reputation of the Advisor
Research Fit with Harvard's Interests -> Harvard Hiring Criteria
Reputation of the Advisor -> External Recommendations
Reputation of the Advisor -> Harvard Hiring Criteria
External Recommendations -> Harvard Hiring Criteria
Harvard Hiring Criteria -> Job Offer at Harvard

"""

dotenv.load_dotenv()
client = OpenAI()



def get_gpt_response(prompt, model="gpt-4o", system_prompt = None, response_format=None):
    """
        sends the prompt to openai
    """
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
        seed = 42, 
        response_format=response_format
    )
    return response.choices[0].message.content


def generate_causal_hypothesis(question):
    prompt = f"""
    Given the following question about career choices:
    "{question}"
    
    1. Identify the key variables involved in this decision.
    2. Hypothesize causal relationships between these variables.
    3. Describe potential confounding factors.
    4. Suggest a causal graph structure in the following format:
       Variable1 -> Variable2
       Variable1 -> Variable3
       Variable2 -> Outcome
       ...
    
    Provide your response in a structured format.
    """

    system_prompt = "You are a causal inference expert tasked with generating hypotheses about causal mechanisms in career decisions."
    response = get_gpt_response(prompt,system_prompt=system_prompt)
    return response



def parse_question_and_hypothesis(question, hypothesis):
    prompt = f"""
    Given the following question and generated hypothesis about a career decision:

    Question: {question}

    Hypothesis: {hypothesis}

    Please identify the following:
    1. The outcome variable (what the decision is trying to achieve)
    2. The choices or treatments being considered (usually two alternatives)
    3. Any other relevant variables mentioned

    Provide your answer in the following JSON format:
    {{
        "outcome": "string",
        "choices": ["string", "string"],
        "other_variables": ["string", "string", ...] }}

    Ensure that the outcome and choices are clearly distinct and directly related to the question.
    """

    system_prompt = "You are an AI assistant specialized in parsing causal inference questions and hypotheses."
    
    response = get_gpt_response(prompt, model="gpt-4o", system_prompt=system_prompt, response_format={"type": "json_object"})

    try :
        parsed_data = json.loads(response)
        return parsed_data
    except json.JSONDecodeError:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            parsed_data = match.group(1).strip()
            print("response post parsing\n" + parsed_data)
            return parsed_data
        else:
            print("Error: GPT's response was not valid JSON. Raw response:")
            print(response)
            return None



def parse_causal_graph(hypothesis):
    edges = []
    for line in hypothesis.split('\n'):
        if '->' in line:
            source, target = line.split('->')
            edges.append((source.strip(), target.strip()))
    return edges

def create_graph(edges):
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR')
    for source, target in edges:
        graph.edge(source, target)
    return graph


def simulate_scenario(G, edge_strengths, num_simulations=1000):
    outcomes = []
    for _ in range(num_simulations):
        node_values = {node: np.random.normal() for node in G.nodes()}
        for edge in G.edges():
            source, target = edge
            node_values[target] += edge_strengths[edge] * node_values[source]
        outcomes.append(node_values['Outcome'])
    return np.mean(outcomes), np.std(outcomes)

def create_interactive_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Create a Pyvis network
    net = Network(notebook=True, width="100%", height="700px", directed=True)
    
    # Add nodes with improved styling
    for node in G.nodes():
        net.add_node(node, label=node, shape="box", size=40, font={"size": 26}, color="#E6F3FF", borderWidth=2)
    
    # Add edges with improved styling
    for edge in G.edges():
        net.add_edge(edge[0], edge[1], arrows="to", color="#4682B4", width=2)
    
    # Configure options for hierarchical layout with improved spacing and selection behavior
    layout_options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "levelSeparation": 250,
                "nodeSpacing": 250,
                "treeSpacing": 300,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
                "direction": "LR",
                "sortMethod": "directed"
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 300,
                "springConstant": 0.01,
                "nodeDistance": 250,
                "damping": 0.09
            },
            "minVelocity": 0.75,
            "solver": "hierarchicalRepulsion"
        },
        "interaction": {
            "dragNodes": True,
            "zoomView": True,
            "dragView": True,
            "hover": True,
            "selectConnectedEdges": True
        },
        "edges": {
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "vertical",
                "roundness": 0.5
            },
            "length": 300,
            "color": {
                "inherit": False,
                "color": "#4682B4",
                "highlight": "#FF0000",  # Bright red for highlighted edges
                "opacity": 1.0
            },
            "width": 2,
            "selectionWidth": 4  # Width of selected edges
        },
        "nodes": {
            "font": {
                "size": 18
            }
        },
        "selection": {
            "highlighting": {
                "enabled": True
            },
            "inheritColors": False
        },
        "highlightNearest": {
            "enabled": True,
            "degree": 1,
            "hover": True
        }
    }
    
    # Convert options to JSON string
    options_json = json.dumps(layout_options)
    
    # Set options using the JSON string
    net.set_options(options_json)
    
    return net

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Adjust to the appropriate model
    )
    return response.data[0].embedding

def map_variables(hypothesis_vars, data_vars, threshold=0.6):
    mapping = {}
    data_embeddings = {var: get_embedding(var) for var in data_vars}
    
    for h_var in hypothesis_vars:
        h_embedding = get_embedding(h_var)
        best_match = None
        best_score = threshold
        
        for d_var, d_embedding in data_embeddings.items():
            score = cosine_similarity([h_embedding], [d_embedding])[0][0]
            if score > best_score:
                best_match = d_var
                best_score = score
        
        if best_match:
            mapping[h_var] = best_match
    
    return mapping

def verify_hypothesis_correlation(edges, data, threshold=0.6):
    validations = []
    hypothesis_vars = set(sum(edges, ()))  # Get unique variables from edges
    data_vars = data.columns.tolist()
    
    # Map hypothesis variables to data variables
    variable_mapping = map_variables(hypothesis_vars, data_vars, threshold)
    
    for source, target in edges:
        if source in variable_mapping and target in variable_mapping:
            mapped_source = variable_mapping[source]
            mapped_target = variable_mapping[target]

            if mapped_source and mapped_target:
                if data[mapped_source].dtype == 'object':
                    data[mapped_source] = pd.factorize(data[mapped_source])[0].astype(float)
                else:
                    data[mapped_source] = data[mapped_source].astype(float)

                if data[mapped_target].dtype == 'object':
                    data[mapped_target] = pd.factorize(data[mapped_target])[0].astype(float)
                else:
                    data[mapped_target] = data[mapped_target].astype(float)

                # Calculate correlation and p-value
                correlation, p_value = stats.pearsonr(data[mapped_source], data[mapped_target])
                validations.append({
                    'edge': f"{source} -> {target}",
                    'mapped_edge': f"{mapped_source} -> {mapped_target}",
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    return validations

def apply_mapping(edges, mapping):
    mapped_edges = []
    for source, target in edges:
        mapped_source = mapping.get(source, source)
        mapped_target = mapping.get(target, target)
        if mapped_source and mapped_target:
            mapped_edges.append((mapped_source, mapped_target))
    return mapped_edges

def learn_causal_graph(data):
    G, edges = pc(data.values, alpha=0.05, indep_test=fisherz, uc_rule=0, verbose=True)
    edges_list = [(data.columns[i], data.columns[j]) for i, j in edges]
    return G, edges_list


def verify_hypothesis_regression(edges, data):
    validations = []
    for source, target in edges:
        if source in data.columns and target in data.columns:
            X = data[[source]]
            y = data[target]
            model = LinearRegression().fit(X, y)
            r_squared = model.score(X, y)
            validations.append({
                'edge': f"{source} -> {target}",
                'r_squared': r_squared,
                'coefficient': model.coef_[0],
                'significant': r_squared > 0.1  # This is a simplification
            })
    return validations


def gpt_variable_mapping(dag_variables, data_features):
    prompt = f"""
    Task: Map variables from a causal DAG to features in a dataset.

    Causal DAG Variables:
    {', '.join(dag_variables)}

    Available Data Features:
    {', '.join(data_features)}

    Instructions:
    1. For each DAG variable, find the best matching or most suitabl    2. Each data feature can be mapped to at most one DAG variable.
    3. If there's no suitable match or proxy for a DAG variable, do not include it in the mapping.
    4. Provide a brief explanation for each mapping or why a variable couldn't be mapped.

    Output the result as a JSON object with the following structure:
    {{
        "mappings": [
            {{"dag_variable": "DAG_VAR1", "data_feature": "DATA_FEATURE1", "explanation": "Reason for mapping"}}, {{"dag_variable": "DAG_VAR2", "data_feature": null, "explanation": "Reason for no mapping"}}
        ]
    }}

    Ensure the output is valid JSON.
    """

    # Get GPT's response
    response = get_gpt_response(prompt, model="gpt-4o", system_prompt="You are an AI assistant skilled in causal inference and data analysis, that only outputs in json format.", response_format={ "type": "json_object" })
    print("response pre parsing\n" + response)
    # Parse the JSON response

    try :
        mapping_result = json.loads(response)
        return mapping_result['mappings']
    except json.JSONDecodeError:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        

        if match:
            mapping_result = match.group(1).strip()
            print("response post parsing\n" + mapping_result)
            return mapping_result['mappings']
        else:
            print("No JSON content found.")


def estimate_causal_effects(data, edges, outcome, choices):
    G = nx.DiGraph(edges)
    effects = {}
    
    for treatment in choices:
        if treatment != outcome:
            # Identify parents of treatment and outcome (potential confounders)
            treatment_parents = list(G.predecessors(treatment))
            outcome_parents = list(G.predecessors(outcome))
            confounders = list(set(treatment_parents + outcome_parents) - set([treatment, outcome]))
            
            # Estimate propensity scores
            X = data[confounders]
            y = (data[treatment] > data[treatment].median()).astype(int)  # Binarize treatment
            propensity_model = LogisticRegression()
            propensity_model.fit(X, y)
            propensity_scores = propensity_model.predict_proba(X)[:, 1]
            
            # Matching
            treated = data[y == 1]
            control = data[y == 0]
            
            treated_outcomes = []
            control_outcomes = []
            
            for i, treated_unit in treated.iterrows():
                treated_ps = propensity_scores[i]
                best_match = min(control.index, key=lambda j: abs(propensity_scores[j] - treated_ps))
                
                treated_outcomes.append(treated_unit[outcome])
                control_outcomes.append(control.loc[best_match, outcome])
            
            # Calculate treatment effect
            effect = np.mean(treated_outcomes) - np.mean(control_outcomes)
            
            # Conduct t-test
            t_stat, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
            
            effects[treatment] = {
                'effect': effect,
                'p_value': p_value,
                'confounders': confounders
            }
    
    return effects


def bootstrap_effect_estimates(data, edges, outcome, n_iterations=1000):
    bootstrap_effects = {node: [] for node in nx.DiGraph(edges).nodes() if node != outcome}
    
    for _ in range(n_iterations):
        bootstrap_sample = data.sample(n=len(data), replace=True)
        effects = estimate_causal_effects(bootstrap_sample, edges, outcome)
        for node, effect in effects.items():
            bootstrap_effects[node].append(effect['effect'])
    
    reliability_results = {}
    for node, effects in bootstrap_effects.items():
        mean_effect = np.mean(effects)
        ci_lower, ci_upper = np.percentile(effects, [2.5, 97.5])
        reliability_results[node] = {
            'mean_effect': mean_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return reliability_results

def generate_recommendation(causal_effects, reliability_results, outcome, choices):
    # Filter effects related to choices (assuming choices are binary and mutually exclusive)
    choice_effects = {k: v for k, v in causal_effects.items() if 'choice' in k.lower()}
    
    if not choice_effects:
        return "No clear choices identified in the causal graph."
    
    # Find the choice with the strongest positive effect
    best_choice = max(choice_effects, key=lambda k: choice_effects[k]['effect'])
    best_effect = choice_effects[best_choice]['effect']
    best_p_value = choice_effects[best_choice]['p_value']
    
    # Get reliability information for the best choice
    reliability = reliability_results.get(best_choice, {})
    ci_lower = reliability.get('ci_lower', None)
    ci_upper = reliability.get('ci_upper', None)
    
    # Generate recommendation
    recommendation = f"Based on the estimated causal effects, the recommended choice is: {best_choice}\n\n"
    recommendation += f"Estimated effect on {outcome}: {best_effect:.4f}\n"
    recommendation += f"P-value: {best_p_value:.4f}\n"
    
    if ci_lower is not None and ci_upper is not None:
        recommendation += f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})\n\n"
    
    # Add interpretation
    if best_p_value < 0.05:
        recommendation += "This effect is statistically significant at the 0.05 level.\n"
    else:
        recommendation += "However, this effect is not statistically significant at the 0.05 level. "
        recommendation += "Consider gathering more data or re-evaluating the causal model.\n"
    
    # Compare to alternatives
    alternatives = [k for k in choice_effects if k != best_choice]
    for alt in alternatives:
        alt_effect = choice_effects[alt]['effect']
        recommendation += f"\nCompared to {alt}:"
        recommendation += f" The recommended choice has an effect {best_effect - alt_effect:.4f} higher on {outcome}.\n"
    
    # Add caveats
    recommendation += "\nCaveats:\n"
    recommendation += "- This recommendation is based on estimated causal effects and should be considered alongside other factors.\n"
    recommendation += "- The analysis assumes the causal graph is correct and all relevant confounders have been accounted for.\n"
    recommendation += "- Individual experiences may vary, and past performance doesn't guarantee future results.\n"
    
    return recommendation


def generate_synthetic_data_from_graph(edges, choices, sample_size=1000):
    G = nx.DiGraph(edges)
    data = {}
    
    # Topological sort to ensure we generate parent variables first
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        if not parents:
            # Root node, generate random data
            data[node] = np.random.normal(0, 1, sample_size)
        else:
            # Generate data based on parents
            parent_data = np.column_stack([data[parent] for parent in parents])
            coeffs = np.random.uniform(0.1, 0.5, len(parents))
            data[node] = np.dot(parent_data, coeffs) + np.random.normal(0, 0.5, sample_size)
    
    # Ensure the choices are included in the data
    for choice in choices:
        if choice not in data:
            data[choice] = np.random.choice([0, 1], sample_size)
    
    return pd.DataFrame(data)
