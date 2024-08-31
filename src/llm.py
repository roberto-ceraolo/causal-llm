# llm.py

import json
import re
from openai import OpenAI
from typing import Dict, Any, List
from config import MODEL
from utils import get_cached_response, cache_response

client = OpenAI()


def get_embedding(text: str) -> List[float]:
    """
    Get the embedding for the given text.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_gpt_response(prompt: str, model: str = MODEL, system_prompt: str = None, response_format: Dict[str, Any] = None) -> str:
    """
    Get a response from the GPT model for the given prompt.

    Args:
    prompt (str): The prompt for which the response is being generated.
    model (str): The GPT model to use.
    system_prompt (str): The system prompt, if any.
    response_format (dict): The format of the response.

    Returns:
    str: The response generated by the GPT model.
    """
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


def generate_initial_dag(question: str) -> str:
    """
    Generates an initial DAG based on the given question and returns JSON.

    Args:
    question (str): The question for which the DAG is being generated.

    Returns:
    str: A JSON string containing the initial DAG data.
    """
    
    prompt = f"""
    Given the following question:
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

    Ensure that all variables used in the causal_graph are defined in the variable_definitions. BE CAREFUL: the graph should be acyclic, so avoid introducing cycles in the graph.
    """
    system_prompt = "You are a causal inference expert tasked with generating hypotheses about causal mechanisms in career decisions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    response_format = {"type": "json_object"}
    return get_gpt_response(prompt, system_prompt=system_prompt, response_format=response_format)

def parse_llm_dag_output(output: str) -> Dict[str, Any]:
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
        

def refine_dag_pc(question: str, refined_dag: List[List[str]], pc_graph: List[List[str]], treatment_variable: str, outcome_variable: str) -> str:
    """
    Additionally refines the DAG by considering the PC graph.
    
    Args:
    initial_dag (dict): A dictionary containing the DAG. 
    pc_graph (list): A list of edges in the PC graph.
    
    Returns:
    str: A JSON string containing the refined DAG output.
    """

    
    pc_additional_prompt = "The following is the causal graph obtained using the PC algorithm on a relevant dataset. Consider this graph in addition to the initial causal graph when refining the DAG."
    for edge in pc_graph:
        source, destination = edge
        pc_additional_prompt += f"\n- {source} -> {destination}"
    
    
    prompt = f"""
    Given the following causal graph obtained via reasoning for the question "{question}":

    {json.dumps(refined_dag, indent=2)}

    The treatment variable is "{treatment_variable}" and the outcome variable is "{outcome_variable}".

    Please refine this graph by considering the PC graph derived from observational data.
    
    {pc_additional_prompt}

    You should not take the PC graph as definitive, but rather as a source of additional information to refine the causal graph. Only make changes to the graph if you have strong reasons to do so based on the PC graph.
    If no changes are necessary, you can keep the graph as is. 
    Based on these considerations, provide your response as a JSON object with the following structure:

    {{
        "causal_graph": [
            ["ShortName1", "ShortName2"],
            ["ShortName1", "ShortName3"],
            ["ShortName2", "Outcome"],
            ...
        ],
        "explanations": "Your explanation of the reasoning behind the changes and additions to the graph, addressing the considerations mentioned above."
    }}

    Ensure that all variables used in the causal_graph were defined in the provided DAG. BE CAREFUL: the graph should be acyclic, so avoid introducing cycles in the graph.
    """
    system_prompt = "You are a causal inference expert tasked with refining and improving causal graphs for career decisions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    response_format = {"type": "json_object"}
    return get_gpt_response(prompt, system_prompt=system_prompt, response_format=response_format)




def refine_dag(question: str, initial_dag: Dict[str, Any]) -> str:
    """
    Refines the initial DAG by considering additional factors and perspectives.
    
    Args:
    initial_dag (dict): A dictionary containing the initial DAG data.
    
    Returns:
    str: A JSON string containing the refined DAG output.
    """

    prompt = f"""
    Given the following initial causal graph for the question "{question}":

    {json.dumps(initial_dag, indent=2)}

    Please refine this graph by considering the following:
    1. Put yourself in the shoes of the decision-maker. What additional factors might they consider that aren't represented in the current graph?
    2. Consider potential long-term consequences that might not be immediately apparent.
    3. Think about external factors (e.g., economic conditions, technological advancements) that could influence the decision or its outcomes.
    4. Identify any feedback loops or bidirectional relationships that might exist.
    5. Consider any hidden variables that might be influencing multiple observed variables.
    6. Could there be any common causes of both the treatment and outcome variables that are not currently included in the graph?



    Based on these considerations, provide your response as a JSON object with the following structure. Please define all the variables that you decide to include in the causal graph, both the ones already present and any new variables you introduce. 

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

    Ensure that all variables used in the causal_graph are defined in the variable_definitions. BE CAREFUL: the graph should be acyclic, so avoid introducing cycles in the graph.
    The treatment and the outcome variables need to be nodes in the graph. 
    """
    system_prompt = "You are a causal inference expert tasked with refining and improving causal graphs for career decisions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    response_format = {"type": "json_object"}
    return get_gpt_response(prompt, system_prompt=system_prompt, response_format=response_format)


def interpret_causal_effect(question: str, treatment_variable: str, outcome_variable: str, causal_estimate: float) -> str:
    prompt = f"""
    Given the following question about causal effects:
    "{question}"

    And given that we estimated the causal effect of {treatment_variable} on {outcome_variable} to be {causal_estimate}, please interpret these results in plain language, explaining what they mean in the context of the original question.

    Answer the question in the best way possible, considering the finding of the estimate but also your reasoning and knowledge about the problem.
    """

    system_prompt = "You are an expert in causal inference tasked with interpreting complex causal effects for a general audience. Provide clear, concise explanations that relate directly to the original question. You want to be as helpful as possible, leveraging your reasoning capbitilies, together with the provided estimate."
    
    return get_gpt_response(prompt, system_prompt=system_prompt)