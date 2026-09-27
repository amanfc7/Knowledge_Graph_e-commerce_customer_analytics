from data_loader import load_data
from build_knowledge_graph import build_graph
from kg_analysis import run_analysis

from ml_node2vec import run_node2vec
from nlp_sentiment import run_sentiment_analysis
from seller_analysis import run_seller_analysis
from geo_analysis import run_geo_analysis

from kg_visualizer import export_graph_to_json
from kg_show_visualization import build_visualization
from data_model_comparison import run_data_model_comparison
from kg_reasoning import run_kg_reasoning

from graph_metrics import run_graph_metrics
from graph_queries import run_graph_queries
from embedding_evaluation import evaluate_embeddings


def main():
    print(" KG PIPELINE STARTING ")

    # 1. load source data
    print("\n--- Loading Data... ---")

    (
        customers,
        orders,
        order_items,
        products,
        payments,
        sellers,
        reviews,
        geo,
        category,
    ) = load_data()

    # 2. build the initial knowledge graph
    print("\n--- BUILDING KNOWLEDGE GRAPH ---")

    G = build_graph(
        customers,
        orders,
        order_items,
        products,
        payments,
        sellers,
        reviews,
        geo,
        category,
    )

    print(f"\nInitial KG nodes: {G.number_of_nodes():,}")
    print(f"Initial KG edges: {G.number_of_edges():,}")

    # 3. data model comparison
    print("\n--- DATA MODEL COMPARISON ---")
    run_data_model_comparison()

    # 4. business analytics
    print("\n--- BUSINESS ANALYTICS ---")

    run_analysis(
        customers,
        orders,
        order_items,
        products,
        payments,
    )

    # 5. NLP-based KG enrichment
    print("\n--- NLP KG ENRICHMENT ---")

    nodes_before_nlp = G.number_of_nodes()
    edges_before_nlp = G.number_of_edges()

    run_sentiment_analysis(
        reviews,
        G,
    )

    print("\nNLP KG evolution:")
    print(
        f"Nodes: {nodes_before_nlp:,} → "
        f"{G.number_of_nodes():,}"
    )
    print(
        f"Edges: {edges_before_nlp:,} → "
        f"{G.number_of_edges():,}"
    )

    # 6. seller analysis
    print("\n--- SELLER ANALYSIS ---")

    run_seller_analysis(
        order_items,
        payments,
    )

    # 7. geographic KG enrichment
    print("\n--- GEO ENRICHMENT ---")

    run_geo_analysis(
        geo,
        G,
    )

    print(
        f"\nKG after enrichment: "
        f"{G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    # 8. logical KG reasoning
    print("\n--- KG LOGICAL REASONING ---")

    run_kg_reasoning(
        G,
    )

    print(
        f"\nKG after reasoning: "
        f"{G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    # 9. final KG validation
    print("\n--- FINAL KG METRICS AND VALIDATION ---")

    run_graph_metrics(
        G,
    )

    # 10. Node2Vec representation of the evolved KG
    print("\n--- NODE2VEC REPRESENTATION ---")

    model = run_node2vec(
        G,
    )

    evaluate_embeddings(
        model,
        G,
    )

    # 11. analytical query service
    print("\n--- KG ANALYTICAL QUERY SERVICE ---")

    run_graph_queries(
        G,
        interactive=False,
    )

    # 12. export the final evolved KG
    print("\n--- EXPORTING KNOWLEDGE GRAPH ---")

    export_graph_to_json(
        G,
    )

    # 13. visualization
    print("\n--- BUILDING VISUALIZATION ---")

    build_visualization()

    # complete
    print("\n==============================")
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("==============================")


if __name__ == "__main__":
    main()