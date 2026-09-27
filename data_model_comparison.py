import os
import pandas as pd

# configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REFLECTION_DIR = os.path.join(RESULTS_DIR, "reflection")


def save_comparison(dataframe, filename):
    os.makedirs(REFLECTION_DIR, exist_ok=True)
    path = os.path.join(REFLECTION_DIR, filename)
    dataframe.to_csv(path, index=False)
    print(f"\nData-model comparison saved to:\n{path}")


# LO4 — data model comparison
def run_data_model_comparison():
    print("\n==============================")
    print(" LO4 — KG DATA MODEL COMPARISON")
    print("==============================")

    # comparison dimensions
    rows = [
        {
            "data_model": "Relational database",
            "primary_representation": "Tables, rows and foreign keys",
            "relationships": "Foreign-key relationships and joins",
            "schema_style": "Schema-first",
            "query_style": "SQL",
            "multi_hop_relationships": "Possible through joins",
            "graph_traversal": "Not native",
            "flexibility": "More structured for tabular data",
            "fit_for_this_project": "Useful for the original Olist tabular datasets",
        },
        {
            "data_model": "RDF / Semantic Web",
            "primary_representation": "Subject-predicate-object triples",
            "relationships": "Predicates",
            "schema_style": "Ontology/vocabulary based",
            "query_style": "SPARQL",
            "multi_hop_relationships": "Supported through graph pattern matching",
            "graph_traversal": "Native graph model",
            "flexibility": "Strong semantic interoperability",
            "fit_for_this_project": (
                "Useful if formal ontology modelling and linked-data "
                "interoperability are priorities"
            ),
        },
        {
            "data_model": "Property graph",
            "primary_representation": "Nodes, relationships and properties",
            "relationships": "Typed relationships with properties",
            "schema_style": "Flexible / schema-optional",
            "query_style": "Graph traversal and graph query APIs",
            "multi_hop_relationships": "Natural graph traversal",
            "graph_traversal": "Native graph representation",
            "flexibility": "High",
            "fit_for_this_project": (
                "Chosen representation for e-commerce relationship "
                "analysis using NetworkX"
            ),
        },
        {
            "data_model": "Document database",
            "primary_representation": "Documents and nested fields",
            "relationships": "References or embedded documents",
            "schema_style": "Flexible",
            "query_style": "Document-oriented queries",
            "multi_hop_relationships": "Usually requires additional lookups or application logic",
            "graph_traversal": "Not a native graph operation",
            "flexibility": "High for document-shaped data",
            "fit_for_this_project": (
                "Useful for document-oriented records but less natural "
                "for multi-hop relationship analysis"
            ),
        },
    ]

    comparison = pd.DataFrame(rows)

    print("\n--- DATA MODEL COMPARISON ---")
    print(comparison.to_string(index=False))
    save_comparison(comparison, "data_model_comparison.csv")

    # project-specific model
    project_rows = [
        {
            "project_requirement": "Customer → Order",
            "chosen_model": "Property graph style",
            "representation": "Customer -[PLACED]-> Order",
        },
        {
            "project_requirement": "Order → Product",
            "chosen_model": "Property graph style",
            "representation": "Order -[CONTAINS]-> Product",
        },
        {
            "project_requirement": "Product → Seller",
            "chosen_model": "Property graph style",
            "representation": "Product -[OFFERED_BY]-> Seller",
        },
        {
            "project_requirement": "Product → Category",
            "chosen_model": "Property graph style",
            "representation": "Product -[BELONGS_TO]-> Category",
        },
        {
            "project_requirement": "Order → Payment",
            "chosen_model": "Property graph style",
            "representation": "Order -[HAS_PAYMENT]-> Payment",
        },
        {
            "project_requirement": "Order → Review",
            "chosen_model": "Property graph style",
            "representation": "Order -[HAS_REVIEW]-> Review",
        },
        {
            "project_requirement": "Review → Sentiment",
            "chosen_model": "Property graph style",
            "representation": "Review -[HAS_SENTIMENT]-> Sentiment",
        },
    ]

    project_model = pd.DataFrame(project_rows)

    print("\n--- PROJECT DATA MODEL ---")
    print(project_model.to_string(index=False))
    save_comparison(project_model, "project_data_model.csv")

    # model decision summary
    decision = pd.DataFrame(
        [
            {
                "aspect": "Primary goal",
                "decision": (
                    "Represent and analyse relationships among customers, "
                    "orders, products, sellers, payments and reviews."
                ),
            },
            {
                "aspect": "Chosen representation",
                "decision": "Property-graph-style representation implemented with NetworkX.",
            },
            {
                "aspect": "Main reason",
                "decision": (
                    "The project requires direct multi-hop relationship "
                    "exploration and analytical graph queries."
                ),
            },
            {
                "aspect": "Query technology",
                "decision": "NetworkX-based analytical query service.",
            },
            {
                "aspect": "Construction technology",
                "decision": "NetworkX graph construction with node and edge attributes.",
            },
            {
                "aspect": "ML representation layer",
                "decision": (
                    "The property graph is also used as input to Node2Vec "
                    "for unsupervised graph representation learning."
                ),
            },
            {
                "aspect": "Representation-learning connection",
                "decision": (
                    "Graph structure is transformed into node embeddings "
                    "that support similarity analysis; Node2Vec is a graph "
                    "embedding method, not a graph neural network."
                ),
            },
            {
                "aspect": "Main alternative",
                "decision": (
                    "RDF could provide formal semantic-web modelling, "
                    "ontology-based representation and linked-data "
                    "interoperability."
                ),
            },
            {
                "aspect": "Trade-off",
                "decision": (
                    "The chosen approach prioritises practical graph "
                    "traversal, analytical querying, representation learning "
                    "and application development rather than formal ontology interoperability."
                ),
            },
        ]
    )

    print("\n--- MODEL DECISION ---")
    print(decision.to_string(index=False))
    save_comparison(decision, "model_decision.csv")

    print("\nLO4 DATA MODEL COMPARISON COMPLETED")


# main
if __name__ == "__main__":
    run_data_model_comparison()