import streamlit as st
import graphviz
from openai import OpenAI
import dotenv
import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from utils import generate_causal_hypothesis, gpt_variable_mapping, parse_causal_graph, create_graph, create_interactive_graph, simulate_scenario, default_q, cached_answer, map_variables, apply_mapping, learn_causal_graph, verify_hypothesis_correlation, verify_hypothesis_regression, parse_question_and_hypothesis, generate_synthetic_data_from_graph, estimate_causal_effects, bootstrap_effect_estimates, generate_recommendation
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt


st.title("Causal Hypothesis Generator and Editor for Career Decisions")

question = st.text_input("Enter your career decision question:", default_q)

if 'edges' not in st.session_state:
    st.session_state.edges = []

if 'hypothesis_generated' not in st.session_state:
    st.session_state.hypothesis_generated = False

if st.button("Generate Hypothesis"):
    if question:
        with st.spinner("Generating causal hypothesis..."):
            hypothesis = generate_causal_hypothesis(question)
            
        st.subheader("Generated Hypothesis:")
        st.text(hypothesis)

        edges = parse_causal_graph(hypothesis)
        st.session_state.edges = edges

        parsed_data = parse_question_and_hypothesis(question, hypothesis)

        st.session_state.hypothesis_generated = True
    
        st.subheader("Parsed Information:")
        st.write(f"Outcome: {parsed_data['outcome']}")
        st.write(f"Choices: {', '.join(parsed_data['choices'])}")
        st.write(f"Other variables: {', '.join(parsed_data['other_variables'])}")

        # Generate synthetic data
        synthetic_data = generate_synthetic_data_from_graph(edges, parsed_data['choices'])
        st.subheader("Generated Synthetic Data:")
        st.write(synthetic_data.head())
        
        # Estimate causal effects
        causal_effects = estimate_causal_effects(synthetic_data, edges, parsed_data['outcome'], parsed_data['choices'])
        
        st.subheader("Estimated Causal Effects:")
        for choice, effect in causal_effects.items():
            st.write(f"Effect of {choice} on {parsed_data['outcome']}: {effect['effect']:.4f} (p-value: {effect['p_value']:.4f})")
            st.write(f"  Confounders considered: {', '.join(effect['confounders'])}")
        
        # Estimate reliability
        reliability_results = bootstrap_effect_estimates(synthetic_data, edges, parsed_data['outcome'], parsed_data['choices'])
        
        st.subheader("Reliability of Causal Effect Estimates:")
        for choice, result in reliability_results.items():
            st.write(f"Effect of {choice} on {parsed_data['outcome']}:")
            st.write(f"  Mean effect: {result['mean_effect']:.4f}")
            st.write(f"  95% CI: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
        
        # Generate recommendation
        recommendation = generate_recommendation(causal_effects, reliability_results, parsed_data['outcome'], parsed_data['choices'])
        
        st.subheader("Recommendation:")
        st.write(recommendation)
 
        st.rerun()
    else:
        st.warning("Please enter a question.")

if st.button("Use cached hypothesis"):
    
    st.subheader("Loaded Hypothesis:")
    st.text(cached_answer)
    st.session_state.edges = parse_causal_graph(cached_answer)
    st.session_state.hypothesis_generated = True
    st.success("Loaded cached hypothesis.")
    st.rerun()


if st.session_state.hypothesis_generated:
    # Visualize graph
    #st.subheader("Visualized Causal Graph:")
    #graph = create_graph(st.session_state.edges)
    #st.graphviz_chart(graph)


    # Initialize the edit_graph state if it doesn't exist
    if 'edit_graph' not in st.session_state:
        st.session_state.edit_graph = False

    # Toggle button for edit graph section
    if st.button("Toggle Edit Graph"):
        st.session_state.edit_graph = not st.session_state.edit_graph

    # Edit graph section
    if st.session_state.edit_graph:
        st.subheader("Edit Causal Graph:")
        
        # Add new edge
        col1, col2 = st.columns(2)
        with col1:
            new_source = st.text_input("New edge source:")
        with col2:
            new_target = st.text_input("New edge target:")
        if st.button("Add Edge"):
            if new_source and new_target:
                st.session_state.edges.append((new_source, new_target))
                st.success(f"Added edge: {new_source} -> {new_target}")
            else:
                st.warning("Please enter both source and target for the new edge.")
        
        # Remove edge
        if st.session_state.edges:
            edge_to_remove = st.selectbox("Select edge to remove:",
                                        [f"{source} -> {target}" for source, target in st.session_state.edges])
            if st.button("Remove Edge"):
                source, target = edge_to_remove.split(" -> ")
                st.session_state.edges.remove((source, target))
                st.success(f"Removed edge: {edge_to_remove}")
    else:
        st.info("Click 'Toggle Edit Graph' to edit the causal graph.")
        
        # Update graph visualization
        #st.subheader("Updated Causal Graph:")
        #updated_graph = create_graph(st.session_state.edges)
        #st.graphviz_chart(updated_graph)
    
    # Interactive Graph
    st.subheader("Interactive Causal Graph:")
    net = create_interactive_graph(st.session_state.edges)
    html = net.generate_html()
    components.html(html, height=600)
    

    # Data Integration and Hypothesis Validation
    st.subheader("Data Integration and Hypothesis Validation")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Only perform mapping if it hasn't been done yet
        if 'mapped_edges' not in st.session_state:
            dag_variables = set([var for edge in st.session_state.edges for var in edge])
            data_features = set(data.columns)
            mappings = gpt_variable_mapping(dag_variables, data_features)
            
            if mappings:
                st.subheader("Variable Mapping Results:")
                for mapping in mappings:
                    st.write(f"DAG Variable: {mapping['dag_variable']}")
                    st.write(f"Data Feature: {mapping['data_feature'] if mapping['data_feature'] else 'No suitable match'}")
                    st.write(f"Explanation: {mapping['explanation']}")
                
                # Create a dictionary for easy lookup
                mapping_dict = {m['dag_variable']: m['data_feature'] for m in mappings if m['data_feature']}
                
                # Apply mapping to edges
                mapped_edges = [(mapping_dict.get(source, source), mapping_dict.get(target, target))
                                for source, target in st.session_state.edges
                                if mapping_dict.get(source) and mapping_dict.get(target)]
                
                # Store mapped_edges in session state
                st.session_state.mapped_edges = mapped_edges
        #else:
        #    st.error("Failed to generate variable mapping. Please try again.")
        #    st.session_state.mapped_edges = []

    if st.button("Validate Hypothesis"):
        if 'mapped_edges' in st.session_state and st.session_state.mapped_edges:
            correlation_validations = verify_hypothesis_correlation(st.session_state.mapped_edges, data)
            regression_validations = verify_hypothesis_regression(st.session_state.mapped_edges, data)
            
            st.subheader("Correlation-based Validation:")
            for v in correlation_validations:
                st.write(f"Edge: {v['edge']}")
                st.write(f"Correlation: {v['correlation']:.2f}")
                st.write(f"P-value: {v['p_value']:.4f}")
                st.write(f"Significant: {'Yes' if v['significant'] else 'No'}")
                st.write("---")
            
            st.subheader("Regression-based Validation:")
            for v in regression_validations:
                st.write(f"Edge: {v['edge']}")
                st.write(f"R-squared: {v['r_squared']:.2f}")
                st.write(f"Coefficient: {v['coefficient']:.2f}")
                st.write(f"Significant: {'Yes' if v['significant'] else 'No'}")
                st.write("---")
        else:
            st.error("Please upload a file and ensure variable mapping is completed before validating the hypothesis.")

        # # PC Algorithm for learning causal graph
        # st.subheader("Learned Causal Graph using PC Algorithm")
        # G, learned_edges = learn_causal_graph(data)

        # # Display learned edges
        # st.write("Learned Edges:")
        # for edge in learned_edges:
        #     st.write(f"{edge[0]} -> {edge[1]}")

        # # Plot the learned causal graph
        # GraphUtils.plot_skeleton(G)
        # st.pyplot(plt)

        # # Compare learned edges with manually created edges
        # st.subheader("Comparison of Hypothesis Edges with Learned Edges")
        # manual_edges = set(mapped_edges)
        # learned_edges_set = set(learned_edges)
        
        # common_edges = manual_edges & learned_edges_set
        # manual_only_edges = manual_edges - learned_edges_set
        # learned_only_edges = learned_edges_set - manual_edges

        # st.write("Common Edges:")
        # for edge in common_edges:
        #     st.write(f"{edge[0]} -> {edge[1]}")

        # st.write("Manual Only Edges:")
        # for edge in manual_only_edges:
        #     st.write(f"{edge[0]} -> {edge[1]}")

        # st.write("Learned Only Edges:")
        # for edge in learned_only_edges:
        #     st.write(f"{edge[0]} -> {edge[1]}")

st.sidebar.markdown("""
## How to use this app:
1. Enter your career decision question in the text box.
2. Click on "Generate Hypothesis" to create a causal hypothesis.
3. The app will display the generated hypothesis and a visualization of the causal graph.
4. Click "Edit Graph" to modify the causal relationships.
5. Use the interactive graph to explore the causal relationships.
6. Upload a CSV file with relevant data to validate the hypothesis.
7. The app will also learn a causal graph from the uploaded data using the PC algorithm and compare it with the manually created graph.

                    
Note: This is a simplified model and should not be the sole basis for making important career decisions.
""")