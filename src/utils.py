# utils.py

import hashlib
import json
import os
from config import CACHE_FILE

def load_cache():
    """ 
    Load the cache from a file.

    Returns:
        dict: The cache loaded from the file.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """
    Save the cache to a file.

    Args:
        cache (dict): The cache to be saved.

    Returns:
        None
    """
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def hash_input(input_str: str) -> str:
    """
    Hash the input string using MD5.

    Args:
        input_str (str): The input string to be hashed.

    Returns:
        The MD5 hash of the input string
    """
    return hashlib.md5(input_str.encode()).hexdigest()

def get_cached_response(prompt: str, system_prompt: str = None) -> str:
    """
    Get the cached response for the given prompt.

    Args:
        prompt (str): The prompt for which the response is being cached.
        system_prompt (str): The system prompt, if any.

    Returns:
        The cached response, if any
    """
    cache = load_cache()
    key = hash_input(prompt + (system_prompt or ""))
    return cache.get(key)

def cache_response(prompt: str, system_prompt: str, response: str):
    """
    Cache the response for the given prompt.
    
    Args:
        prompt (str): The prompt for which the response is being cached.
        system_prompt (str): The system prompt, if any.
        response (str): The response to be cached.
    
    Returns:
        None
    """
    cache = load_cache()
    key = hash_input(prompt + (system_prompt or ""))
    cache[key] = response
    save_cache(cache)
    