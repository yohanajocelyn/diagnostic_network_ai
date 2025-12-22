import os
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import numpy as np
# Importing from your project structure
from network_structure import load_and_train_model 

# 1. FIXED: Added double underscores to __file__
# Points to the data folder relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'data', 'dataset.csv')

def visualize_network_in_streamlit():
    try:
        # 2. Load model
        model, _ = load_and_train_model(csv_path)

        # Create a specific figure object
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 3. FIXED: Added 'iterations' and 'seed' to ensure consistent layout
        # shell_layout is good, but spring_layout shows the "Web" better
        pos = nx.spring_layout(model, k=1.5, iterations=100, seed=42)

        # 4. FIXED: The "Jitter" fix to prevent the StopIteration/Bezier error
        # This ensures no two nodes have the exact same coordinates
        for node in pos:
            pos[node] += np.random.uniform(-0.02, 0.02, size=2)

        # Draw the nodes
        nx.draw_networkx_nodes(
            model, pos, 
            node_size=2500, 
            node_color="#FF9999", 
            edgecolors="black", # Added border for a cleaner look
            ax=ax
        )

        # Draw the edges (Arrows)
        nx.draw_networkx_edges(
            model, pos, 
            edge_color="gray", 
            arrowstyle='-|>', 
            arrowsize=25, 
            width=1.5,
            connectionstyle='arc3, rad = 0.1', # Curved edges like your reference
            ax=ax
        )

        # Draw Labels
        nx.draw_networkx_labels(
            model, pos, 
            font_size=9, 
            font_weight="bold", 
            ax=ax
        )

        ax.set_title("Learned Causal Structure (Bayesian Network)", fontsize=15)
        ax.axis('off') 
        
        # Display in Streamlit
        st.pyplot(fig) 

    except Exception as e:
        st.error(f"An error occurred generating the graph: {e}")

# 5. FIXED: Added double underscores to __name__ and __main__
if __name__ == "__main__":
    visualize_network_in_streamlit()