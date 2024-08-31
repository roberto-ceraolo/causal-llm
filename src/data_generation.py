# data_generation.py

import numpy as np
import pandas as pd

def generate_random_dataset(variables: set, outcome: str, treatment: str, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a random dataset with the given variables. The outcome is generated based on the other variables.

    Args:
        variables: Set of variables to generate the dataset.
        outcome: The outcome variable.
        treatment: The treatment variable.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Random dataset with the given variables.
    """
    data = {}
    
    np.random.seed(seed)

    for var in variables:
        # Default to continuous (0, 1) for unknown variables
        if var == outcome:
            continue
        if var == treatment:
            data[var] = np.random.randint(0, 2, n_samples)
        else:
            data[var] = np.random.random(n_samples)
    df = pd.DataFrame(data)
    
    # Generate Outcome based on other variables if it's in the variables
    if outcome in variables:
        predictors = [var for var in variables if var != outcome]
        coeffs = np.random.uniform(-1, 1, len(predictors))
        linear_combination = np.sum([coeff * df[var] for coeff, var in zip(coeffs, predictors)], axis=0)
        probability = 1 / (1 + np.exp(-linear_combination))
        df[outcome] = (np.random.random(n_samples) < probability).astype(int)
    
    return df
