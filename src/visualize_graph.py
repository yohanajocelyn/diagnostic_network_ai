import streamlit as st
import networkx as nx
from network_structure import load_and_train_model # Ensure this import matches your file structure
import os

# Define path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'data', 'dataset.csv')

def visualize_network_in_streamlit():
    st.title("Diagnostic Logic Map")
    
    try:
        # 1. Load the Model
        with st.spinner("Learning Structure..."):
            model, _ = load_and_train_model(csv_path)

        # 2. Convert NetworkX to Graphviz (DOT format) manually
        # This avoids installing heavy dependencies like 'pydot' on Windows
        dot_code = 'digraph G {\n'
        
        # A. Global Graph Settings to match your image style
        dot_code += '  rankdir=TB;\n'       # TB = Top to Bottom layout
        dot_code += '  splines=ortho;\n'    # Use Right-Angle lines (like the image)
        dot_code += '  nodesep=0.4;\n'      # Space between nodes
        dot_code += '  ranksep=0.8;\n'      # Space between layers
        
        # B. Node Styling (Rectangular Boxes)
        dot_code += '  node [shape=box, style=filled, fillcolor="#f9f9f9", fontname="Sans-Serif"];\n'
        
        # C. Add Edges from your Learned Model
        for u, v in model.edges():
            # Sanitize names (replace spaces with underscores if needed for internal IDs)
            # But keep readable labels
            dot_code += f'  "{u}" -> "{v}";\n'
            
        dot_code += '}'

        # 3. Render directly in Streamlit
        st.graphviz_chart(dot_code, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing graph: {e}")

if __name__ == "__main__":
    visualize_network_in_streamlit()