# Causal Inference for Natural Causal Questions

This project implements a tool-augmented AI agent that can answer natural causal questions. It uses a combination of causal inference techniques, LLM, and tools to answer questions. It can provide support in decision making, in scientific research, and in policy making.

## Features

1. Generation of initial causal graphs based on user questions
2. Refinement of causal graphs using expert knowledge and data-driven insights
3. Integration with OpenAI's GPT models for natural language processing
4. Caching system for efficient response retrieval
5. Synthetic data generation for testing and development
6. Kaggle dataset integration for real-world data analysis

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Set up your Kaggle API credentials (optional, for Kaggle dataset functionality)

## Usage

The main functionality is implemented in the `src/causal_engine.py` file. Key functions include:

- `generate_initial_dag`: Generates an initial causal graph based on a user question
- `refine_dag`: Refines the causal graph by considering additional factors
- `refine_dag_pc`: Further refines the graph using insights from the PC algorithm
- `interpret_causal_effect`: Interprets the causal effect estimates





## Kaggle Integration

The project includes functionality to search for and use relevant datasets from Kaggle:

- `find_and_prepare_kaggle_dataset`: Searches for a relevant Kaggle dataset based on the causal question and prepares it for analysis
- `build_and_save_embeddings`: Builds and saves embeddings for Kaggle datasets to improve search efficiency
- `load_embeddings`: Loads pre-computed embeddings for Kaggle datasets

To use Kaggle functionality, ensure you have set up your Kaggle API credentials and set `KAGGLE = True` in the configuration.

## Configuration

The project configuration is stored in `src/config.py`. You can modify settings such as:

- `CACHE_FILE`: Location of the cache file for LLM responses
- `MODEL`: The GPT model to use
- `SYNTHETIC_DATA`: Whether to use synthetic data generation
- `DEBUG`: Enable or disable debug mode
- `KAGGLE`: Enable or disable Kaggle dataset functionality

## Contributing

Contributions to this project are welcome. Please ensure that you follow the existing code style and add appropriate tests for new features.


## Acknowledgements

This project uses OpenAI's GPT models for natural language processing and causal inference tasks, as well as the Kaggle API for dataset retrieval.