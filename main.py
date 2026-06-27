from data_loader import load_data
from build_knowledge_graph import build_graph
from kg_analysis import run_analysis

import matplotlib.pyplot as plt
import networkx as nx

def main():

    # Step 1: Load data
    customers, orders, order_items, products, payments, category = load_data()

    # Step 2: Build graph
    G = build_graph(customers, orders, order_items, products, payments, category)

    print("\nGraph built successfully!")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # Step 3: Analysis
    run_analysis(customers, orders, order_items, products, payments)

    # Step 4: Visualization (small sample)
    sample_nodes = list(G.nodes())[:100]
    subgraph = G.subgraph(sample_nodes)

    plt.figure(figsize=(10, 6))
    nx.draw(subgraph, node_size=20, with_labels=False)
    plt.title("E-commerce Knowledge Graph (Sample)")
    plt.show()

if __name__ == "__main__":
    main()