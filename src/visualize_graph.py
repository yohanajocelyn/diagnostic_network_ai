import os
import matplotlib.pyplot as plt
import networkx as nx
# Ensure your pathing is correct for your 'src' folder
from network_structure import load_and_train_model

# Path to your dataset
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')

def visualize():
    try:
        model, _ = load_and_train_model(csv_path)

        plt.figure(figsize=(14, 10))
        
        # Spring layout is better for showing 'web' structures
        pos = nx.spring_layout(model, k=1.5, seed=42)

        nx.draw(
            model, pos,
            with_labels=True,
            node_size=2000,
            node_color="#FF9999", # Light red to distinguish from the previous star
            font_size=9,
            font_weight="bold",
            edge_color="gray",
            arrowsize=20,
            width=1.5
        )

        plt.title("Learned Web Structure (Hill Climbing Search)")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    visualize()