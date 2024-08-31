from typing import Dict, List
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
import openai
import numpy as np
import faiss
import pickle
from openai import OpenAI
from llm import get_embedding




from llm import get_gpt_response, parse_llm_dag_output

api = KaggleApi()
api.authenticate()

def get_kaggle_dataset(keyword):
    # List datasets based on the keyword
    list_datasets = api.datasets_list(search=keyword)
    
    if not list_datasets:
        return None
    
    # Take the first dataset
    dataset_ref = list_datasets[0]["ref"]
    
    # Extract the dataset name from the ref
    dataset_name = dataset_ref.split("/")[-1]
    
    # Download the dataset
    api.dataset_download_files(dataset_ref)
    
    # Create a directory with the same name as the dataset
    os.makedirs(dataset_name, exist_ok=True)
    
    # Unzip the dataset
    with zipfile.ZipFile(f"{dataset_name}.zip", "r") as zip_ref:
        zip_ref.extractall(dataset_name)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(f"{dataset_name}/data.csv")
    
    return df


def map_variables(dag_variables: List[str], dataset_columns: List[str]) -> Dict[str, str]:
    prompt = f"""
    Given the following variables from a causal DAG:
    {', '.join(dag_variables)}

    And the following columns from a dataset:
    {', '.join(dataset_columns)}

    Please map the DAG variables to the most appropriate dataset columns. 
    If there's no suitable match for a DAG variable, use 'None'.
    Provide your answer as a Python dictionary where keys are DAG variables and values are dataset columns or 'None'.
    """
    
    response = get_gpt_response(prompt)

    #response = parse_llm_dag_output(response)


    # Assuming the response is a valid Python dictionary string
    return eval(response)

"""
keyword version

def find_and_prepare_kaggle_dataset(treatment: str, outcome: str, dag_variables: List[str]) -> pd.DataFrame:
    keyword = f"{treatment}"
    df = get_kaggle_dataset(keyword)
    
    if df is None:
        return None
    
    variable_mapping = map_variables(dag_variables, df.columns.tolist())
    
    # Filter and rename columns based on the mapping
    relevant_columns = [col for col in variable_mapping.values() if col != 'None']
    df_filtered = df[relevant_columns]
    
    rename_dict = {v: k for k, v in variable_mapping.items() if v != 'None'}
    df_renamed = df_filtered.rename(columns=rename_dict)
    
    return df_renamed

"""


def generate_dataset_embedding(dataset_info):
    text = f"{dataset_info['title']} {dataset_info['subtitle']} {dataset_info['description']} "
    text += f"Tags: {', '.join([tag['name'] for tag in dataset_info['tags']])}"
    
    return get_embedding(text)

def build_and_save_embeddings(api: KaggleApi, save_path: str = 'kaggle_embeddings'):
    datasets = api.datasets_list(page=1)
    embeddings = []
    dataset_info = []
    
    for dataset in datasets:
        embedding = generate_dataset_embedding(dataset)
        embeddings.append(embedding)
        dataset_info.append({
            'ref': dataset['ref'],
            'title': dataset['title'],
            'subtitle': dataset['subtitle'],
            'url': dataset['url'],
            'description': dataset['description'],
            'tags': [tag['name'] for tag in dataset['tags']]
        })
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Build FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Save FAISS index
    faiss.write_index(index, f'{save_path}.index')
    
    # Save dataset info
    with open(f'{save_path}_info.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)

def load_embeddings(load_path: str = 'kaggle_embeddings') -> tuple:
    # Load FAISS index
    index = faiss.read_index(f'{load_path}.index')
    
    # Load dataset info
    with open(f'{load_path}_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    return index, dataset_info

def find_similar_dataset(query: str, index: faiss.Index, dataset_info: List[Dict]) -> Dict:

    query_embedding = get_embedding(query)

    # Convert query embedding to numpy array
    query_array = np.array([query_embedding]).astype('float32')

    # Perform similarity search
    k = 1  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_array, k)

    return dataset_info[indices[0][0]]

def find_and_prepare_kaggle_dataset(treatment: str, outcome: str, dag_variables: List[str], index: faiss.Index, dataset_info: List[Dict]) -> pd.DataFrame:
    query = f"Dataset for analyzing the effect of {treatment} on {outcome}, considering variables: {', '.join(dag_variables)}"
    
    similar_dataset = find_similar_dataset(query, index, dataset_info)
    
    dataset_ref = similar_dataset['ref']

    print(f"Found similar dataset: {dataset_ref}")

    print(f"Dataset description: {similar_dataset['description']}")
    
    api.dataset_download_files(dataset_ref)
    
    dataset_name = dataset_ref.split("/")[-1]
    
    with zipfile.ZipFile(f"{dataset_name}.zip", "r") as zip_ref:
        zip_ref.extractall(dataset_name)
    
    df = pd.read_csv(f"{dataset_name}/data.csv")
    
    variable_mapping = map_variables(dag_variables, df.columns.tolist())
    
    relevant_columns = [col for col in variable_mapping.values() if col != 'None']
    df_filtered = df[relevant_columns]
    
    rename_dict = {v: k for k, v in variable_mapping.items() if v != 'None'}
    df_renamed = df_filtered.rename(columns=rename_dict)
    
    return df_renamed



# Build and save embeddings (run this once)
#build_and_save_embeddings(api, num_datasets=1000)

