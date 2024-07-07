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

import pandas as pd
import numpy as np

def generate_random_dataset(variables: set, n_samples: int = 1000):
    data = {}
    for var in variables:
        # Determine variable type based on name (you can modify this logic as needed)
        if var.lower().endswith(('degree', 'outcome')):
            data[var] = np.random.choice([0, 1], n_samples)
        elif var.lower().endswith(('skills', 'networking', 'opportunities')):
            data[var] = np.random.randint(1, 6, n_samples)
        elif var.lower().endswith('experience'):
            data[var] = np.random.randint(0, 21, n_samples)
        else:
            # Default to continuous (0, 1) for unknown variables
            data[var] = np.random.uniform(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Generate Outcome based on other variables if it's in the variables
    if 'Outcome' in variables:
        predictors = [var for var in variables if var != 'Outcome']
        coeffs = np.random.uniform(-1, 1, len(predictors))
        linear_combination = np.sum([coeff * df[var] for coeff, var in zip(coeffs, predictors)], axis=0)
        probability = 1 / (1 + np.exp(-linear_combination))
        df['Outcome'] = (np.random.random(n_samples) < probability).astype(int)
    
    return df

def get_gpt_response(prompt: str, model: str = "gpt-4o", system_prompt: str = None, response_format: Dict[str, Any] = None) -> str:
    """
    Sends the prompt to OpenAI and returns the response.
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
        seed=42,
        response_format=response_format
    )
    return response.choices[0].message.content

def generate_initial_dag(question: str, cached: bool = False) -> str:
    """
    Generates an initial DAG based on the given question and returns JSON.
    """
    if cached: 
        return '{\n    "variable_definitions": {\n        "GradDegree": "Pursuing a graduate degree",\n        "JobOpportunities": "Availability of job opportunities",\n        "Networking": "Networking opportunities during and after the graduate program",\n        "WorkExperience": "Work experience prior to or during the graduate program",\n        "Skills": "Skills and knowledge acquired during the graduate program",\n        "Income": "Long-term income",\n        "JobSatisfaction": "Long-term job satisfaction",\n        "EconomicConditions": "Overall economic conditions",\n        "PersonalMotivation": "Personal motivation and career aspirations",\n        "Outcome": "Long-term career prospects"\n    },\n    "causal_graph": [\n        ["GradDegree", "Skills"],\n        ["GradDegree", "Networking"],\n        ["GradDegree", "JobOpportunities"],\n        ["Skills", "JobOpportunities"],\n        ["Networking", "JobOpportunities"],\n        ["JobOpportunities", "Outcome"],\n        ["WorkExperience", "JobOpportunities"],\n        ["WorkExperience", "Skills"],\n        ["EconomicConditions", "JobOpportunities"],\n        ["EconomicConditions", "Outcome"],\n        ["PersonalMotivation", "GradDegree"],\n        ["PersonalMotivation", "Outcome"],\n        ["Income", "Outcome"],\n        ["JobSatisfaction", "Outcome"]\n    ],\n    "explanations": "The causal graph illustrates the hypothesized relationships between pursuing a graduate degree and long-term career prospects. Pursuing a graduate degree (GradDegree) is expected to directly enhance Skills and Networking opportunities, which in turn increase JobOpportunities. These job opportunities are a direct contributor to the long-term career prospects (Outcome). WorkExperience is included as it can influence both Skills and JobOpportunities. EconomicConditions are considered a potential confounding factor affecting both JobOpportunities and the Outcome. PersonalMotivation is another confounder influencing both the decision to pursue a graduate degree and the long-term career prospects. Long-term career prospects (Outcome) are also influenced by Income and JobSatisfaction, which are outcomes of JobOpportunities. Assumptions made include that pursuing a graduate degree generally leads to better skills and networking, which are crucial for job opportunities, and that economic conditions and personal motivation are significant external factors."\n}'
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

def refine_dag(initial_dag: Dict[str, Any], cached: bool = False) -> str:
    """
    Refines the initial DAG by considering additional factors and perspectives.
    
    Args:
    initial_dag (dict): A dictionary containing the initial DAG data.
    
    Returns:
    str: A JSON string containing the refined DAG output.
    """
    if cached: 
        return '{\n    "variable_definitions": {\n        "GradDegree": "Pursuing a graduate degree",\n        "JobOpportunities": "Availability of job opportunities",\n        "Networking": "Networking opportunities during and after the graduate program",\n        "WorkExperience": "Work experience prior to or during the graduate program",\n        "Skills": "Skills and knowledge acquired during the graduate program",\n        "Income": "Long-term income",\n        "JobSatisfaction": "Long-term job satisfaction",\n        "EconomicConditions": "Overall economic conditions",\n        "PersonalMotivation": "Personal motivation and career aspirations",\n        "Outcome": "Long-term career prospects",\n        "TechnologicalAdvancements": "Impact of technological advancements on the job market",\n        "FamilySupport": "Support from family and friends",\n        "Health": "Physical and mental health",\n        "JobMarketTrends": "Trends in the job market",\n        "Debt": "Debt incurred from pursuing a graduate degree"\n    },\n    "causal_graph": [\n        ["GradDegree", "Skills"],\n        ["GradDegree", "Networking"],\n        ["GradDegree", "JobOpportunities"],\n        ["GradDegree", "Debt"],\n        ["Skills", "JobOpportunities"],\n        ["Networking", "JobOpportunities"],\n        ["JobOpportunities", "Outcome"],\n        ["WorkExperience", "JobOpportunities"],\n        ["WorkExperience", "Skills"],\n        ["EconomicConditions", "JobOpportunities"],\n        ["EconomicConditions", "Outcome"],\n        ["PersonalMotivation", "GradDegree"],\n        ["PersonalMotivation", "Outcome"],\n        ["Income", "Outcome"],\n        ["JobSatisfaction", "Outcome"],\n        ["TechnologicalAdvancements", "JobOpportunities"],\n        ["TechnologicalAdvancements", "Skills"],\n        ["FamilySupport", "GradDegree"],\n        ["FamilySupport", "Outcome"],\n        ["Health", "GradDegree"],\n        ["Health", "Outcome"],\n        ["JobMarketTrends", "JobOpportunities"],\n        ["JobMarketTrends", "Outcome"],\n        ["Debt", "Outcome"],\n        ["Debt", "Income"]\n    ],\n    "explanations": "The refined causal graph incorporates several additional factors that a decision-maker might consider. \'Debt\' is added to capture the financial burden of pursuing a graduate degree, which can influence both \'Outcome\' and \'Income\'. \'TechnologicalAdvancements\' and \'JobMarketTrends\' are included to account for external factors that can affect job opportunities and skills. \'FamilySupport\' and \'Health\' are added to represent personal factors that can influence both the decision to pursue a graduate degree and long-term career prospects. The bidirectional relationship between \'Debt\' and \'Income\' is acknowledged, as debt can impact income and vice versa. Additionally, feedback loops are considered, such as the impact of \'TechnologicalAdvancements\' on both \'Skills\' and \'JobOpportunities\'. These changes provide a more comprehensive view of the factors influencing career decisions and their long-term consequences."\n}'

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
        "explanations": "Your explanation of the reasoning behind the changes and additions to the graph, addressing the considerations mentioned above."
    }}

    Ensure that all variables used in the causal_graph are defined in the variable_definitions.
    """
    system_prompt = "You are a causal inference expert tasked with refining and improving causal graphs for career decisions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    response_format = {"type": "json_object"}
    return get_gpt_response(prompt, system_prompt=system_prompt, response_format=response_format)



def formalize_query(question, refined_dag, cached=False):
    """
    Formalizes the query into a mathematical expression.
    """
    if cached: 
        return '```json\n{\n    "formalized_query": "E[Outcome | do(GradDegree = 1)] - E[Outcome | do(GradDegree = 0)]",\n    "reasoning": "The query asks about the effect of pursuing a graduate degree on long-term career prospects. In causal inference, this is typically expressed as the average treatment effect (ATE) of the treatment \'GradDegree\' on the outcome \'Outcome\'. The \'do(·)\' notation is used to denote interventions. Specifically, \'do(GradDegree = 1)\' represents the intervention where an individual pursues a graduate degree, and \'do(GradDegree = 0)\' represents the intervention where an individual does not pursue a graduate degree. To find the effect, we need to compare the expected outcomes under these two interventions. Therefore, the formalized query is the difference in the expected value of \'Outcome\' when \'GradDegree\' is set to 1 versus when \'GradDegree\' is set to 0."\n}\n```'


    prompt = f"""
    Given the following question about career choices and the causal graph:

    Question: {question}

    Edges of the Causal Graph:
    {refined_dag}

    Please formalize the query by translating it into its formal mathematical expression. Use the "do(·)" notation or counterfactual notations as needed. Explain your reasoning step by step.
    Provide your response as a JSON object with the following structure:
        {{
            "reasoning": "Your step-by-step reasoning and calculations",
            "formalized_query": "Your formalized query"
        }}
    """
    system_prompt = "You are a causal inference expert tasked with formalizing queries into mathematical expressions. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    return get_gpt_response(prompt, system_prompt=system_prompt)

def deduce_estimand(formalized_query, refined_dag, cached=False):
    """
    Deduces the estimand using causal inference techniques.
    """
    if cached: 
        return '{\n    "reasoning": "To deduce the estimand E[Outcome | do(GradDegree = 1)] - E[Outcome | do(GradDegree = 0)], we need to identify the causal effect of GradDegree on Outcome. The given causal graph includes multiple paths and potential confounders. We will use do-calculus to adjust for these confounders. The backdoor paths from GradDegree to Outcome are through PersonalMotivation, FamilySupport, Health, and Debt. We need to block these paths to isolate the direct effect of GradDegree on Outcome. Additionally, we need to account for mediators like Skills, Networking, and JobOpportunities. Using the backdoor adjustment formula, we can express the causal effect as follows: E[Outcome | do(GradDegree = g)] = Σ_skills Σ_networking Σ_jobopportunities P(Outcome | GradDegree = g, Skills = s, Networking = n, JobOpportunities = j) * P(Skills = s | GradDegree = g) * P(Networking = n | GradDegree = g) * P(JobOpportunities = j | Skills = s, Networking = n, GradDegree = g, WorkExperience, EconomicConditions, TechnologicalAdvancements, JobMarketTrends). This accounts for the direct and indirect effects through the mediators. Finally, we compute the difference between the two potential outcomes: E[Outcome | do(GradDegree = 1)] - E[Outcome | do(GradDegree = 0)].",\n    "estimand": "Σ_skills Σ_networking Σ_jobopportunities [P(Outcome | GradDegree = 1, Skills = s, Networking = n, JobOpportunities = j) * P(Skills = s | GradDegree = 1) * P(Networking = n | GradDegree = 1) * P(JobOpportunities = j | Skills = s, Networking = n, GradDegree = 1, WorkExperience, EconomicConditions, TechnologicalAdvancements, JobMarketTrends)] - Σ_skills Σ_networking Σ_jobopportunities [P(Outcome | GradDegree = 0, Skills = s, Networking = n, JobOpportunities = j) * P(Skills = s | GradDegree = 0) * P(Networking = n | GradDegree = 0) * P(JobOpportunities = j | Skills = s, Networking = n, GradDegree = 0, WorkExperience, EconomicConditions, TechnologicalAdvancements, JobMarketTrends)]"\n}'


    prompt = f"""
    Given the following formalized query and Causal Graph:

    Formalized Query: {formalized_query}

    Edges of the Causal Graph:
    {refined_dag}

    Please deduce the estimand using causal inference techniques. Your approach should utilize skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Provide a step-by-step explanation of your reasoning and calculations.
    Provide your response as a JSON object with the following structure:
        {{
            "reasoning": "Your step-by-step reasoning and calculations",
            "estimand": "Your deduced estimand formula"
        }}
    """
    system_prompt = "You are a causal inference expert tasked with deducing estimands using advanced causal inference techniques. Provide your output as a valid JSON object. Do not include any other character except the JSON object. Don't include the markdown syntax anywhere."
    return get_gpt_response(prompt, system_prompt=system_prompt)

class EstimandParser:
    def __init__(self, estimand_string: str, data: pd.DataFrame = None, n_samples: int = 1000):
        self.estimand_string = estimand_string
        self.variables = self._extract_variables()
        if data is None:
            self.data = generate_random_dataset(self.variables, n_samples)
        else:
            self.data = data
        self.parsed_estimand = self._parse_estimand()

    def _extract_variables(self) -> set:
        pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b'
        return set(re.findall(pattern, self.estimand_string))

    def _parse_estimand(self):
        # Replace probability notation with sympy's P function
        ######################################### TODO - FIX THE ESTIMAND PARSER #########################################
        #expr = self.estimand_string.replace('P(', 'P(')
        expr = self.estimand_string
        expr_sympy = self.convert_to_sympy(expr)
        
        # Replace summation notation with sympy's Sum
        sum_pattern = r'Σ_(\w+)'
        sums = re.findall(sum_pattern, expr)
        sum_var_map = {}
        for sum_var in sums:
            sum_symbol = Symbol(sum_var.lower())
            sum_var_map[sum_var] = sum_symbol
            expr = expr.replace(f'Σ_{sum_var}', f'Sum({sum_symbol}, ({sum_symbol}, {sum_var}_min, {sum_var}_max))')
        
        # Replace capitalized variables in probability expressions with sum symbols
        for var, symbol in sum_var_map.items():
            expr = re.sub(rf'\b{var}\s*=\s*{var}\b', f'{var} = {symbol}', expr)
        
        # Parse the expression
        return sympify(expr)

    def _get_variable_range(self, variable):
        if self.data[variable].dtype in ['int64', 'float64']:
            return (self.data[variable].min(), self.data[variable].max())
        else:
            return list(self.data[variable].unique())

    def _conditional_probability(self, outcome, conditions):
        mask = pd.Series(True, index=self.data.index)
        for var, value in conditions.items():
            if pd.api.types.is_numeric_dtype(self.data[var]):
                mask &= (self.data[var] == value)
            else:
                mask &= (self.data[var] == value)
        
        if mask.sum() == 0:
            return 0
        
        return (self.data.loc[mask, outcome] == 1).mean()
    
    def _parse_estimand_with_llm(self):
        prompt = f"""
        Given the following estimand string:
        
        {self.estimand_string}

        I have already defined the following symbols: {self.variables}
        
        Please convert this into a SymPy-compatible expression. Follow these guidelines:
        1. Use SymPy's Sum for summations (e.g., Sum(expression, (var, start, end)))
        2. Use P() for probabilities
        3. Ensure all variables are properly symbolized
        4. Use lowercase symbols for summation variables
        5. Ensure consistency between summation variables and their usage in probabilities
        
        Return only the SymPy-compatible expression, without any explanations. Return in json format, with the following structure: 

        {{
            "symbols": "additional symbols, if any, that need to be defined. Not the code, only the string with the names of the symbols, that can be inserted in symbols()",
            "expression": "the string expression in sympy"
        }}
        """
        
        llm_response = get_gpt_response(prompt)
        parsed_llm_response = json.loads(llm_response)
        symbols = parsed_llm_response.get('symbols', '')
        expression = parsed_llm_response.get('expression', '')

        return symbols, expression


    def evaluate(self) -> float:
        def P_func(*args):
            if len(args) == 1:
                return (self.data[args[0]] == 1).mean()
            else:
                outcome = args[0]
                conditions = {k: v for k, v in zip(args[1::2], args[2::2])}
                return self._conditional_probability(outcome, conditions)

        var_ranges = {var: self._get_variable_range(var) for var in self.variables}
        
        subs_dict = {
            P: P_func,
            E: lambda x: np.mean(self.data[x]),
            **{f'{var}_min': min(var_ranges[var]) for var in var_ranges},
            **{f'{var}_max': max(var_ranges[var]) for var in var_ranges},
            **{var: symbols(var) for var in self.variables}
        }

        result = self.parsed_estimand.evalf(subs=subs_dict)
        return float(result)

    def get_variables(self) -> set:
        return self.variables

    def convert_to_sympy(self, expression: str) -> Basic:
        prompt = f"Convert the following mathematical expression to SymPy format:\n\n'{expression}'\n\nProvide the Python SymPy code to represent this expression. Define the needed variables before. Answer ONLY with python code, no markdown or explanation needed."
        
        response = get_gpt_response(prompt)
        
        # Evaluate the response as Python code to create the SymPy expression
        local_vars = {}
        try:
            print(response)
            exec(response, {"sp": sp}, local_vars)
            sympy_expr = local_vars.get('expression')
            if sympy_expr is None:
                raise ValueError("The response did not contain a valid SymPy expression")
            return sympy_expr
        except Exception as e:
            raise ValueError(f"Failed to convert the response to SymPy: {e}")    



def causal_effect_solver(question: str) -> Dict[str, Any]:
    
    # Generate and refine the DAG
    print("Generating initial DAG...")
    initial_dag = generate_initial_dag(question, cached=True)
    parsed_initial_dag = parse_dag_output(initial_dag)
    var_definitions, initial_graph, initial_explanations = parsed_initial_dag['variable_definitions'], parsed_initial_dag['causal_graph'], parsed_initial_dag['explanations']
    print("Initial DAG generated. Variables: ", var_definitions)
    print("Causal Graph: ", initial_graph)
    print("Explanations: ", initial_explanations)

    # Refine the DAG based on additional factors
    print("Refining DAG...")
    refined_output = refine_dag(initial_dag, cached=True)
    parsed_refined_dag = parse_dag_output(refined_output)
    refined_var_definitions, refined_graph, refined_explanations = parsed_refined_dag['variable_definitions'], parsed_refined_dag['causal_graph'], parsed_refined_dag['explanations']
    print("Refined Variable Definitions: ", refined_var_definitions)
    print("DAG refined. Refined DAG: ", refined_graph)
    print("Refined Explanations: ", refined_explanations)

    # Formalize the query and deduce the estimand
    print("Formalizing query...")
    formalized_query = formalize_query(question, refined_graph, cached=True)
    parsed_formalized_query = parse_dag_output(formalized_query)
    final_formalized_query, formalization_reasoning = parsed_formalized_query['formalized_query'], parsed_formalized_query['reasoning']
    print("Query formalized. Formalized query: ", final_formalized_query)
    print("Reasoning: ", formalization_reasoning)

    print("Deducing estimand...")
    estimand = deduce_estimand(final_formalized_query, refined_graph, cached=True)
    parsed_estimand = parse_dag_output(estimand)
    estimand_formula, estimand_reasoning = parsed_estimand['estimand'], parsed_estimand['reasoning']

    print("Estimand deduced. Estimand formula: ", estimand_formula)
    print("Reasoning: ", estimand_reasoning)

    parser = EstimandParser(estimand_formula)

    # Evaluate the estimand
    result = parser.evaluate()
    print(f"Estimand evaluation result: {result}")
    
    return {
        "initial_dag": initial_dag,
        "refined_dag": refined_graph,
        "formalized_query": formalized_query,
        "estimand": estimand,
        "parsed_estimand": parsed_estimand,
        "result": result
    }

dotenv.load_dotenv()
client = OpenAI()
question = "How does pursuing a graduate degree affect long-term career prospects?"
result = causal_effect_solver(question)
print(result)

