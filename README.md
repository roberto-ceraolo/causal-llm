# Causal Solver

This project implements a causal inference framework to answer causal questions. It uses a combination of expert knowledge, causal discovery algorithms, and causal inference techniques to provide insights.

## Features

- Generation and refinement of causal graphs based on expert knowledge
- Integration with PC algorithm for data-driven causal discovery
- Synthetic data generation for testing and demonstration
- Causal effect estimation using DoWhy library
- Caching of LLM responses for efficiency

## Project Structure

```
.
├── config.py
├── utils.py
├── llm.py
├── graph_utils.py
├── causal_discovery.py
├── data_generation.py
├── causal_inference.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/roberto-ceraolo/causal-llm.git
   cd causal-llm
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

To run the causal inference analysis:

```python
python main.py
```

This will execute the causal effect solver for a predefined question. To analyze a different question, modify the `question` variable in `main.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

