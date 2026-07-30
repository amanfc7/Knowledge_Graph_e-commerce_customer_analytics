from pyexpat import model

from data_loader import load_data

from build_knowledge_graph import build_graph
from kg_analysis import run_analysis

from ml_node2vec import run_node2vec

from nlp_sentiment import run_sentiment_analysis
from seller_analysis import run_seller_analysis
from geo_analysis import run_geo_analysis

from kg_visualizer import export_graph_to_json
from kg_show_visualization import build_visualization


# NEW KG EVALUATION MODULES
from graph_metrics import run_graph_metrics
from graph_queries import run_graph_queries
# from community_detection import run_community_detection
from embedding_evaluation import evaluate_embeddings


def main():

    print("\n==============================")
    print(" KGMS PIPELINE STARTING ")
    print("==============================")


    # =====================================================
    # 1. DATA INGESTION LAYER
    # LO1 - Data acquisition
    # LO2 - Data preprocessing
    # =====================================================

    print("\n--- LOADING KG DATA LAYER ---")

    (
        customers,
        orders,
        order_items,
        products,
        payments,
        sellers,
        reviews,
        geo,
        category

    ) = load_data()



    # =====================================================
    # 2. KNOWLEDGE GRAPH CONSTRUCTION
    # LO7 - Knowledge Representation
    # LO8 - Data Integration
    # =====================================================

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
        category
    )


    print("\n--- KG SUMMARY ---")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())



    # =====================================================
    # 3. GRAPH STRUCTURE ANALYSIS
    # LO5 - Graph modelling
    # LO6 - Interpretation
    # =====================================================

    run_graph_metrics(G)



    # =====================================================
    # 4. KG REASONING + BUSINESS ANALYTICS
    # LO4 - Data Analytics
    # LO6 - Insights
    # =====================================================

    run_analysis(
        customers,
        orders,
        order_items,
        products,
        payments
    )



    # =====================================================
    # 5. SYMBOLIC KG QUERIES
    # LO7 - Reasoning over Knowledge Graph
    # =====================================================

    run_graph_queries(G)



    # =====================================================
    # 6. COMMUNITY DISCOVERY
    # Graph-based pattern discovery
    # =====================================================

    # run_community_detection(G)



    # =====================================================
    # 7. MACHINE LEARNING ON GRAPH
    # LO9 - Machine Learning
    # =====================================================

    model = run_node2vec(G)


    evaluate_embeddings(
        model
    )



    # =====================================================
    # 8. NLP KNOWLEDGE ENRICHMENT
    # LO11 - Applied AI
    # =====================================================
    
    run_sentiment_analysis(reviews)



    # =====================================================
    # 9. SELLER ECONOMIC ANALYSIS
    # Financial KG layer
    # =====================================================

    run_seller_analysis(
        order_items,
        payments
    )



    # =====================================================
    # 10. GEO-SPATIAL ANALYSIS
    # =====================================================

    run_geo_analysis(
        geo
    )



    # =====================================================
    # 11. EXPORT KNOWLEDGE GRAPH
    # LO12 - System implementation
    # =====================================================

    export_graph_to_json(
        G
    )



    # =====================================================
    # 12. VISUALIZATION
    # =====================================================

    build_visualization()



    print("\n==============================")
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("==============================")



if __name__ == "__main__":
    main()