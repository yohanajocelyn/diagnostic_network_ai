import streamlit as st
from network_structure import load_and_train_model
import os

# Define path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'data', 'dataset.csv')

def visualize_network_in_streamlit():
    st.title("Diagnostic Logic Map")
    
    try:
        # Load the Model
        with st.spinner("Learning Structure..."):
            model, _ = load_and_train_model(csv_path)

        # Convert NetworkX to Graphviz (DOT format) manually
        dot_code = 'digraph G {\n'
        
        # Settings untuk mengatur tampilan graphnya
        dot_code += '  rankdir=TB;\n'       # TB = Top to Bottom layout
        dot_code += '  splines=ortho;\n'    # Use Right-Angle lines = kalau ada garis yang bengkok, bengkoknya 90 derajat
        dot_code += '  nodesep=0.4;\n'      # Jarak antar node secara horizontal
        dot_code += '  ranksep=0.8;\n'      # Jarak antar layer/node secara vertikal
        
        # Style dari nodenya
        dot_code += '  node [shape=box, style=filled, fillcolor="#f9f9f9", fontname="Sans-Serif"];\n'
        
        # Menambahkan edge atau relationship dari modelnya
        for u, v in model.edges():
            dot_code += f'  "{u}" -> "{v}";\n'
            
        dot_code += '}'

        # Streamlit bisa baca dot code langsung dan render graphnya
        st.graphviz_chart(dot_code, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing graph: {e}")

if __name__ == "__main__":
    visualize_network_in_streamlit()