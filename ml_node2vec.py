import json
import os
import pickle

import numpy as np
import pandas as pd
from node2vec import Node2Vec


# project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Node2Vec configuration
DIMENSIONS = 64
WALK_LENGTH = 20
NUM_WALKS = 30
WINDOW = 10
MIN_COUNT = 1
WORKERS = 1
P = 1.0
Q = 1.0


# ML-based KG evolution configuration
# Only product-to-product similarity relationships are added.
# This keeps the derived knowledge semantically meaningful and bounded.
EVOLUTION_NODE_TYPE = "product"
EVOLUTION_MAX_SOURCES = 500
EVOLUTION_TOP_K = 3
EVOLUTION_SIMILARITY_THRESHOLD = 0.90
EVOLUTION_CANDIDATE_TOP_N = 50
EVOLUTION_RELATION = "EMBEDDING_SIMILAR_TO"


# theory bridge
# LO1 — knowledge graph embeddings
# LO12 — connections between KGs, ML and AI
def explain_embedding_relation():
    print("\n--- THEORY BRIDGE: KG EMBEDDINGS AND GNNS ---")
    print(
        """
Knowledge Graph:
- Nodes represent entities or events.
- Edges represent relationships.

Node2Vec:
- Random walks explore graph neighbourhoods.
- Walk sequences are used with Skip-gram.
- Nodes are represented as numerical vectors.
- Similar graph contexts can produce similar embeddings.

This project uses Node2Vec for unsupervised graph
representation learning.
"""
    )


# save JSON configuration or statistics
def save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return path


# ML-based KG evolution
def evolve_kg_with_embeddings(G, model):
    print(
        "\n--- ML-BASED KNOWLEDGE GRAPH EVOLUTION ---"
    )

    if G is None:
        print(
            "\nEmbedding-based KG evolution skipped: graph is None."
        )
        return 0

    if model is None:
        print(
            "\nEmbedding-based KG evolution skipped: model is None."
        )
        return 0

    # Select only nodes of the chosen semantic type.
    candidate_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("type") == EVOLUTION_NODE_TYPE
        and str(node) in model.wv
    ]

    # Keep the ML evolution bounded for this project.
    candidate_nodes = sorted(
        candidate_nodes,
        key=lambda node: str(node),
    )[:EVOLUTION_MAX_SOURCES]

    print(
        "Evolution node type:",
        EVOLUTION_NODE_TYPE,
    )

    print(
        "Candidate source nodes:",
        len(candidate_nodes),
    )

    print(
        "Similarity threshold:",
        EVOLUTION_SIMILARITY_THRESHOLD,
    )

    print(
        "Top-K similar nodes per source:",
        EVOLUTION_TOP_K,
    )

    print(
        "Maximum candidate neighbours inspected:",
        EVOLUTION_CANDIDATE_TOP_N,
    )

    derived_rows = []
    added_pairs = set()
    added_edges = 0

    for source_node in candidate_nodes:
        source_string = str(source_node)

        try:
            similar_nodes = model.wv.most_similar(
                source_string,
                topn=EVOLUTION_CANDIDATE_TOP_N,
            )
        except Exception:
            continue

        accepted_for_source = 0

        for similar_node, score in similar_nodes:
            similar_node = str(similar_node)
            score = float(score)

            if similar_node == source_string:
                continue

            if score < EVOLUTION_SIMILARITY_THRESHOLD:
                continue

            if similar_node not in G:
                continue

            similar_data = G.nodes[similar_node]

            if similar_data.get("type") != EVOLUTION_NODE_TYPE:
                continue

            # Similarity is conceptually symmetric.
            # Avoid adding both directions of the same derived pair.
            pair_key = frozenset(
                [source_string, similar_node]
            )

            if pair_key in added_pairs:
                continue

            # Do not overwrite or duplicate an existing graph relationship.
            if G.has_edge(source_node, similar_node):
                continue

            if G.has_edge(similar_node, source_node):
                continue

            G.add_edge(
                source_node,
                similar_node,
                relation=EVOLUTION_RELATION,
                weight=score,
                similarity_score=score,
                inferred=True,
                knowledge_source="Node2Vec",
                embedding_method="Node2Vec",
                evolution_type="ML-derived",
            )

            added_pairs.add(pair_key)

            derived_rows.append(
                {
                    "source_node": source_string,
                    "target_node": similar_node,
                    "relation": EVOLUTION_RELATION,
                    "cosine_similarity": score,
                    "embedding_method": "Node2Vec",
                    "knowledge_source": "Node2Vec",
                    "evolution_type": "ML-derived",
                    "inferred": True,
                }
            )

            added_edges += 1
            accepted_for_source += 1

            if accepted_for_source >= EVOLUTION_TOP_K:
                break

    evolution_path = os.path.join(
        RESULTS_DIR,
        "embedding_derived_relationships.csv",
    )

    if derived_rows:
        evolution_df = pd.DataFrame(
            derived_rows
        )

        evolution_df.to_csv(
            evolution_path,
            index=False,
        )

        print(
            "\nSaved embedding-derived relationships:",
            evolution_path,
        )
    else:
        print(
            "\nNo embedding-derived relationships were added."
        )

    evolution_summary = {
        "evolution_method": "Node2Vec embedding similarity",
        "relation_added": EVOLUTION_RELATION,
        "source_node_type": EVOLUTION_NODE_TYPE,
        "candidate_source_nodes": len(candidate_nodes),
        "maximum_sources": EVOLUTION_MAX_SOURCES,
        "top_k_per_source": EVOLUTION_TOP_K,
        "candidate_top_n": EVOLUTION_CANDIDATE_TOP_N,
        "similarity_threshold": EVOLUTION_SIMILARITY_THRESHOLD,
        "derived_relationships_added": added_edges,
        "input_graph_edges_before_evolution": (
            G.number_of_edges() - added_edges
        ),
        "final_graph_edges_after_evolution": (
            G.number_of_edges()
        ),
        "knowledge_source": "Node2Vec",
        "inferred": True,
    }

    evolution_summary_path = save_json(
        evolution_summary,
        "embedding_kg_evolution.json",
    )

    print(
        "\n--- ML KG EVOLUTION SUMMARY ---"
    )

    print(
        "Embedding-derived relationships added:",
        added_edges,
    )

    print(
        "KG edges after ML evolution:",
        G.number_of_edges(),
    )

    print(
        "Saved evolution summary:",
        evolution_summary_path,
    )

    return added_edges


# Node2Vec model
def run_node2vec(G):
    print(
        "\n--- KG EMBEDDINGS: NODE2VEC GRAPH LEARNING ---"
    )

    explain_embedding_relation()

    if G is None:
        print(
            "\nNode2Vec skipped: graph is None."
        )
        return None

    if G.number_of_nodes() == 0:
        print(
            "\nNode2Vec skipped: graph contains no nodes."
        )
        return None

    print("\nInput graph:")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # configuration
    print("\n--- NODE2VEC CONFIGURATION ---")
    print("Dimensions:", DIMENSIONS)
    print("Walk length:", WALK_LENGTH)
    print("Walks per node:", NUM_WALKS)
    print("Window:", WINDOW)
    print("p:", P)
    print("q:", Q)
    print("Workers:", WORKERS)

    # train Node2Vec
    node2vec = Node2Vec(
        G,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=WORKERS,
        p=P,
        q=Q,
    )

    print(
        "\nTraining Node2Vec embedding model..."
    )

    model = node2vec.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=128,
    )

    print(
        "Embedding training completed."
    )

    # save model
    model_path = os.path.join(
        MODELS_DIR,
        "node2vec_model.pkl",
    )

    with open(
        model_path,
        "wb",
    ) as file:
        pickle.dump(
            model,
            file,
        )

    print(
        "Saved embedding model:",
        model_path,
    )

    # save configuration
    configuration = {
        "algorithm": "Node2Vec",
        "representation_type": "graph_embedding",
        "is_gnn": False,
        "learning_type": "unsupervised",
        "dimensions": DIMENSIONS,
        "walk_length": WALK_LENGTH,
        "num_walks": NUM_WALKS,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "workers": WORKERS,
        "p": P,
        "q": Q,
        "input_nodes": G.number_of_nodes(),
        "input_edges": G.number_of_edges(),
        "model_path": model_path,
        "kg_evolution": True,
        "evolution_relation": EVOLUTION_RELATION,
        "evolution_node_type": EVOLUTION_NODE_TYPE,
        "evolution_max_sources": EVOLUTION_MAX_SOURCES,
        "evolution_top_k": EVOLUTION_TOP_K,
        "evolution_similarity_threshold": EVOLUTION_SIMILARITY_THRESHOLD,
    }

    configuration_path = save_json(
        configuration,
        "node2vec_configuration.json",
    )

    print(
        "Saved configuration:",
        configuration_path,
    )

    # embedding statistics
    vocabulary = list(
        model.wv.index_to_key
    )

    if vocabulary:
        vectors = np.array([
            model.wv[node]
            for node in vocabulary
        ])

        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        embedding_statistics = {
            "algorithm": "Node2Vec",
            "representation_type": "graph_embedding",
            "is_gnn": False,
            "embedding_count": len(vocabulary),
            "embedding_dimensions": int(
                model.wv.vector_size
            ),
            "average_vector_norm": float(
                np.mean(norms)
            ),
            "minimum_vector_norm": float(
                np.min(norms)
            ),
            "maximum_vector_norm": float(
                np.max(norms)
            ),
        }
    else:
        embedding_statistics = {
            "algorithm": "Node2Vec",
            "representation_type": "graph_embedding",
            "is_gnn": False,
            "embedding_count": 0,
            "embedding_dimensions": DIMENSIONS,
        }

    statistics_path = save_json(
        embedding_statistics,
        "embedding_statistics.json",
    )

    print(
        "Saved embedding statistics:",
        statistics_path,
    )

    # similarity examples
    print(
        "\n--- LATENT KG REPRESENTATION USING EMBEDDINGS ---"
    )

    sample_nodes = list(
        G.nodes()
    )[:5]

    similarity_rows = []

    for node in sample_nodes:
        node_string = str(node)

        if node_string not in model.wv:
            continue

        print("\nNode:", node)

        similar_nodes = model.wv.most_similar(
            node_string,
            topn=5,
        )

        for similar, score in similar_nodes:
            score = float(score)

            print(
                " ->",
                similar,
                "similarity:",
                round(score, 3),
            )

            similarity_rows.append({
                "source_node": node_string,
                "similar_node": str(similar),
                "cosine_similarity": score,
                "embedding_method": "Node2Vec",
            })

    # save similarity examples
    similarity_path = os.path.join(
        RESULTS_DIR,
        "node_similarity_results.csv",
    )

    if similarity_rows:
        similarity_df = pd.DataFrame(
            similarity_rows
        )

        similarity_df.to_csv(
            similarity_path,
            index=False,
        )

        print(
            "\nSaved similarity results:",
            similarity_path,
        )
    else:
        print(
            "\nNo similarity examples were available."
        )

    return model


# main
if __name__ == "__main__":
    print(
        "This module is executed through main.py."
    )