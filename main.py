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
from kg_show_visualization import build_visualization


def main():

    print("\n==============================")
    print(" KGMS PIPELINE STARTING ")
    print("==============================")

    # -------------------------------------------------
    # 1. DATA LAYER (GROUND FACTS)
    # -------------------------------------------------
    customers, orders, order_items, products, payments, sellers, reviews, geo, category = load_data()

    # -------------------------------------------------
    # 2. KNOWLEDGE GRAPH CONSTRUCTION
    # -------------------------------------------------
    G = build_graph(customers, orders, order_items, products, payments, category)

    print("\n--- KG SUMMARY ---")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # -------------------------------------------------
    # 3. REASONING + ANALYTICS LAYER
    # -------------------------------------------------
    run_analysis(customers, orders, order_items, products, payments)

    # -------------------------------------------------
    # 4. MACHINE LEARNING / EMBEDDINGS (GNN-LIKE)
    # -------------------------------------------------
    run_node2vec(G)

    # -------------------------------------------------
    # 5. NLP LAYER (UNSTRUCTURED DATA)
    # -------------------------------------------------
    run_sentiment_analysis(reviews)

    # -------------------------------------------------
    # 6. SELLER ANALYSIS (ECONOMIC KG LAYER)
    # -------------------------------------------------
    run_seller_analysis(order_items, payments)

    # -------------------------------------------------
    # 7. GEO SPATIAL KG LAYER
    # -------------------------------------------------
    run_geo_analysis(geo)

    # -------------------------------------------------
    # 8. EXPORT KG (FOR VISUALIZATION / ML)
    # -------------------------------------------------
    export_graph_to_json(G)

    # -------------------------------------------------
    # 9. VISUALIZATION
    # -------------------------------------------------
    build_visualization()

    print("\n==============================")
    print(" PIPELINE COMPLETED ")
    print("==============================")


if __name__ == "__main__":
    main()