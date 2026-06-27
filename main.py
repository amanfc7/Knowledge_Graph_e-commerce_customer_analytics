from data_loader import load_data
from build_knowledge_graph import build_graph
from kg_analysis import run_analysis

import matplotlib.pyplot as plt
import networkx as nx

# NEW MODULES
from ml_node2vec import run_node2vec
from nlp_sentiment import analyze_sentiment

def main():

    customers, orders, order_items, products, payments, sellers, reviews, geo, category = load_data()

    G = build_graph(customers, orders, order_items, products, payments, category)

    print("\nGraph built successfully!")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # -------------------
    # ANALYTICS
    # -------------------
    run_analysis(customers, orders, order_items, products, payments)

    # -------------------
    # ML (EXCEED LO)
    # -------------------
    model = run_node2vec(G)

    # -------------------
    # OPTIONAL VISUALIZATION
    # -------------------
    sample_nodes = list(G.nodes())[:100]
    subgraph = G.subgraph(sample_nodes)

    plt.figure(figsize=(10, 6))
    nx.draw(subgraph, node_size=20, with_labels=False)
    plt.title("E-commerce Knowledge Graph")
    plt.show()

if __name__ == "__main__":
    main()