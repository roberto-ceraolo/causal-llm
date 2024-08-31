import gradio as gr
from causal_engine import causal_effect_solver
import networkx as nx
import matplotlib.pyplot as plt
import io

def plot_graph(graph):
    G = nx.DiGraph()
    for edge in graph:
        G.add_edge(edge[0], edge[1])
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', 
            arrows=True, edge_color='gray')
    
    plt.title("Causal Graph")
    plt.axis('off')

    # Convert the plot to a numpy array
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_arr = plt.imread(img_buf)
    plt.close()

    return img_arr
 
    
def process_question(question):
    result = causal_effect_solver(question)
    
    initial_dag = result['initial_dag']["causal_graph"]
    refined_dag = result['refined_dag']["causal_graph"]
    
    identified_estimand = result['identified_estimand']
    initial_dag_graph = plot_graph(initial_dag)
    refined_dag_graph = plot_graph(refined_dag)
    
    inference_method = f"Chosen inference method: {result['estimation_method']}"

    if "causal_estimate" in result:
        if isinstance(result['causal_estimate'], dict):
            causal_estimate = result['causal_estimate'].get('causal_estimate', result['causal_estimate'].get('average_causal_effect', "No estimate found"))
        else:
            causal_estimate = result['causal_estimate']
    else:
        causal_estimate = "No estimate found"

    interpretation = result.get('interpretation', "No interpretation available.")
    
    final_answer = f"Causal estimate: {causal_estimate}\n\nInterpretation: {interpretation}"
    
    return initial_dag_graph, refined_dag_graph, identified_estimand, inference_method, final_answer

iface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your causal question here..."),
    outputs=[
        "image",
        "image",
        gr.Textbox(label="Identified Estimand"),
        gr.Textbox(label="Inference Method"),
        gr.Textbox(label="Final Answer")
    ],
    title="Causal Effect Solver",
    description="Enter a causal question to see the initial and refined DAGs, chosen inference method, and final answer.",
    css="footer{display:none !important}",
    allow_flagging="never"
)

iface.launch()