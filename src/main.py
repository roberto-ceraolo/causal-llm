# main.py

from causal_engine import causal_effect_solver

if __name__ == "__main__":
    #question = "How does pursuing a graduate degree affect long-term career prospects?"
    question = "Should I do a PhD if my goal is having a meaningful impact?"
    result = causal_effect_solver(question) # also a dataset can be provided. 
    print(result)
