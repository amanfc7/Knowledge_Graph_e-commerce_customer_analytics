
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
- Graph structure provides relational context.

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
        "This module is normally executed through main.py."
    )

