import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from causal_engine import causal_effect_solver
from graph_utils import is_dag, create_gml_graph, save_causal_graph_png

def plot_graph(graph, title):
    G = nx.DiGraph(graph)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', 
            arrows=True, arrowsize=20)
    plt.title(title)
    return plt

def explore_data(data):
    st.subheader("Data Exploration")
    
    # Display basic information about the dataset
    st.write("Dataset Shape:", data.shape)
    st.write("Columns:", data.columns.tolist())
    
    # Display first few rows of the dataset
    st.write("First few rows of the dataset:")
    st.dataframe(data.head())
    
    # Display summary statistics
    st.write("Summary Statistics:")
    st.dataframe(data.describe())
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Pairplot for selected variables
    st.subheader("Pairplot")
    selected_vars = st.multiselect("Select variables for pairplot", data.columns.tolist(), default=data.columns.tolist()[:3])
    if selected_vars:
        fig = sns.pairplot(data[selected_vars], height=2.5)
        st.pyplot(fig)
    
    # Histogram for individual variables
    st.subheader("Histogram")
    selected_var = st.selectbox("Select a variable for histogram", data.columns.tolist())
    fig, ax = plt.subplots()
    data[selected_var].hist(bins=30, ax=ax)
    ax.set_title(f"Histogram of {selected_var}")
    st.pyplot(fig)

def add_explainer_sidebar():
    st.sidebar.title("Causal Inference Steps")
    st.sidebar.markdown("""
    1. **Initial DAG Generation**: 
       - Create an initial Directed Acyclic Graph (DAG) based on the input question.
       - Define variables and their relationships.

    2. **DAG Refinement**:
       - Use reasoning to refine the initial DAG.
       - Consider additional factors and potential confounders.

    3. **Data Generation/Loading**:
       - Generate synthetic data or load real data based on the refined DAG.

    4. **PC Algorithm (if data available)**:
       - Apply the PC algorithm to learn the causal structure from data.
       - Further refine the DAG based on PC algorithm results.

    5. **Causal Model Creation**:
       - Create a CausalModel object using the refined DAG and data.

    6. **Effect Identification**:
       - Identify the causal effect using do-calculus and graph structure.

    7. **Estimation Method Selection**:
       - Choose an appropriate estimation method based on the identified estimand and data.

    8. **Causal Effect Estimation**:
       - Estimate the causal effect using the selected method.

    9. **Results Visualization**:
       - Display the initial and refined DAGs.
       - Show the estimated causal effect and other relevant information.

    10. **Data Exploration** (Optional):
        - Explore the dataset used in the analysis through various visualizations and statistics.
    """)

def main():
    st.set_page_config(layout="wide")
    add_explainer_sidebar()

    st.title("Causal Inference Analysis")

    # Input question
    question = st.text_input("Enter your causal inference question:")

    if st.button("Analyze"):
        if question:
            with st.spinner("Analyzing..."):
                # Run causal effect solver
                results, data = causal_effect_solver(question)

                # Display initial DAG
                st.subheader("Initial Causal Graph")
                initial_graph = plot_graph(results['initial_dag'], "Initial Causal Graph")
                st.pyplot(initial_graph)

                # Display refined DAG
                st.subheader("Refined Causal Graph")
                refined_graph = plot_graph(results['refined_dag'], "Refined Causal Graph")
                st.pyplot(refined_graph)

                # Display results
                st.subheader("Causal Effect Analysis Results")
                st.write("Identified Estimand:", results['identified_estimand'])
                st.write("Causal Effect Estimate:", results['estimate'])
                st.write("Estimation Method:", results['method'])

                # Data Exploration
                if data is not None:
                    explore_data(data)
                else:
                    st.write("No dataset available for exploration.")

        else:
            st.warning("Please enter a question to analyze.")

if __name__ == "__main__":
    main()