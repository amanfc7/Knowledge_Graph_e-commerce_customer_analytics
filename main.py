from data_loader import load_data
from build_knowledge_graph import build_graph
from kg_analysis import run_analysis

import matplotlib.pyplot as plt
import networkx as nx

from ml_node2vec import run_node2vec
from nlp_sentiment import run_sentiment_analysis
from seller_analysis import run_seller_analysis
from geo_analysis import run_geo_analysis
from kg_visualizer import export_graph_to_json


def main():

    customers, orders, order_items, products, payments, sellers, reviews, geo, category = load_data()

    # -----------------------
    # BUILD GRAPH
    # -----------------------
    G = build_graph(customers, orders, order_items, products, payments, category)

    print("\nGraph built successfully!")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # -----------------------
    # ANALYTICS (CORE LO)
    # -----------------------
    run_analysis(customers, orders, order_items, products, payments)

    # -----------------------
    # ML (EXCEED LO)
    # -----------------------
    run_node2vec(G)

    # -----------------------
    # NLP (EXCEED LO)
    # -----------------------
    run_sentiment_analysis(reviews)

    # -----------------------
    # SELLER ANALYSIS (EXCEED LO)
    # -----------------------
    run_seller_analysis(order_items, payments)

    # -----------------------
    # GEO ANALYSIS (EXCEED LO)
    # -----------------------
    run_geo_analysis(geo)

    # -----------------------
    # VISUALIZATION
    # -----------------------
    sample_nodes = list(G.nodes())[:100]
    subgraph = G.subgraph(sample_nodes)

    plt.figure(figsize=(10, 6))
    nx.draw(subgraph, node_size=20, with_labels=False)
    plt.title("E-commerce Knowledge Graph")
    plt.show()

    # -----------------------
    # EXPORT
    # -----------------------
    export_graph_to_json(G)


if __name__ == "__main__":
    main()