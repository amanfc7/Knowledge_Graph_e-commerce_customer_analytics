import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# embedding quality evaluation
# KG representation learning analysis
#
# Node2Vec is a graph embedding method that learns low-dimensional vector representations of nodes in a graph.

# This evaluation is designed to assess the quality of the learned embeddings
# in terms of their ability to capture the structural relationships in the graph.

MAX_HOPS = 3
TOP_K = 5
MAX_SAMPLE_SIZE = 300
EVALUATION_SOURCES = 10


def evaluate_embeddings(model, G):
    """Evaluate Node2Vec embeddings and save portfolio-ready results."""
    print("\n--- EMBEDDING QUALITY ANALYSIS ---")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # sample embeddings
    labels = list(model.wv.index_to_key[:MAX_SAMPLE_SIZE])

    if not labels:
        print("No embeddings available for evaluation.")
        return pd.DataFrame(columns=["node", "x", "y"])

    vectors = np.array([model.wv[node] for node in labels])

    print("Embedding dimension:", vectors.shape[1])
    print("Number of sampled nodes:", len(labels))

    # node similarity examples
    print("\n--- NODE SIMILARITY ANALYSIS ---")

    similarity_results = []

    for node in labels[:10]:
        for similar_node, score in model.wv.most_similar(
            node,
            topn=min(TOP_K, max(1, len(labels) - 1)),
        ):
            similarity_results.append({
                "source_node": node,
                "similar_node": similar_node,
                "similarity_score": round(float(score), 4),
            })

            print(
                node,
                "->",
                similar_node,
                ":",
                round(float(score), 4),
            )

    similarity_df = pd.DataFrame(similarity_results)
    similarity_df.to_csv(
        os.path.join(results_dir, "node_similarity_results.csv"),
        index=False,
    )
    print("Saved: results/node_similarity_results.csv")

    # project-defined structural evaluation
    #
    # A similar node is considered structurally related when it is
    # reachable from the source within MAX_HOPS in an undirected view.
    #
    # This produces TP/FP-style examples for the project, but it does
    # not represent ground-truth classification because no labelled
    # positive/negative dataset is used.

    print("\n--- PROJECT-DEFINED STRUCTURAL EVALUATION ---")

    structural_graph = G.to_undirected()
    evaluation_sources = min(EVALUATION_SOURCES, len(labels))
    evaluation_results = []

    for source_node in labels[:evaluation_sources]:
        similar_nodes = model.wv.most_similar(
            source_node,
            topn=min(TOP_K, max(1, len(labels) - 1)),
        )

        reachable_nodes = nx.single_source_shortest_path_length(
            structural_graph,
            source_node,
            cutoff=MAX_HOPS,
        )

        source_type = G.nodes[source_node].get("type", "unknown")

        for rank, (similar_node, score) in enumerate(
            similar_nodes,
            start=1,
        ):
            path_length = reachable_nodes.get(similar_node)

            structurally_related = (
                path_length is not None
                and path_length > 0
                and path_length <= MAX_HOPS
            )

            similar_type = G.nodes[similar_node].get(
                "type",
                "unknown",
            )

            evaluation_results.append({
                "source_node": source_node,
                "source_type": source_type,
                "similar_node": similar_node,
                "similar_type": similar_type,
                "similarity_rank": rank,
                "similarity_score": round(float(score), 4),
                "path_length": (
                    int(path_length)
                    if path_length is not None
                    else None
                ),
                "max_hops": MAX_HOPS,
                "structurally_related": structurally_related,
                "evaluation": (
                    "true_positive"
                    if structurally_related
                    else "false_positive"
                ),
                "evaluation_type": "project_defined_structural_proxy",
            })

    structural_columns = [
        "source_node",
        "source_type",
        "similar_node",
        "similar_type",
        "similarity_rank",
        "similarity_score",
        "path_length",
        "max_hops",
        "structurally_related",
        "evaluation",
        "evaluation_type",
    ]

    structural_df = pd.DataFrame(
        evaluation_results,
        columns=structural_columns,
    )

    structural_output = os.path.join(
        results_dir,
        "embedding_structural_evaluation.csv",
    )
    structural_df.to_csv(structural_output, index=False)

    if structural_df.empty:
        true_positive_count = 0
        false_positive_count = 0
    else:
        true_positive_count = int(
            (structural_df["evaluation"] == "true_positive").sum()
        )
        false_positive_count = int(
            (structural_df["evaluation"] == "false_positive").sum()
        )

    total_predictions = true_positive_count + false_positive_count
    structural_precision = (
        true_positive_count / total_predictions
        if total_predictions
        else 0.0
    )

    print("Evaluation sources:", evaluation_sources)
    print("Top-K per source:", TOP_K)
    print("Structural relationship threshold:", MAX_HOPS, "hops")
    print("Project-defined true positives:", true_positive_count)
    print("Project-defined false positives:", false_positive_count)
    print("Project-defined structural precision:",
          round(structural_precision, 4))
    print("Saved:", structural_output)

    # explicit portfolio examples
    example_parts = []

    if not structural_df.empty:
        true_positive_examples = structural_df[
            structural_df["evaluation"] == "true_positive"
        ].head(3)

        false_positive_examples = structural_df[
            structural_df["evaluation"] == "false_positive"
        ].head(3)

        if not true_positive_examples.empty:
            example_parts.append(true_positive_examples)

        if not false_positive_examples.empty:
            example_parts.append(false_positive_examples)

    if example_parts:
        examples_df = pd.concat(
            example_parts,
            ignore_index=True,
        )
    else:
        examples_df = structural_df.head(5).copy()

    examples_output = os.path.join(
        results_dir,
        "embedding_examples.csv",
    )
    examples_df.to_csv(examples_output, index=False)
    print("Saved:", examples_output)

    # vector statistics
    vector_norms = np.linalg.norm(vectors, axis=1)

    embedding_stats = {
        "embedding_method": "Node2Vec",
        "embedding_dimension": int(vectors.shape[1]),
        "sample_nodes": len(labels),
        "average_vector_norm": float(np.mean(vector_norms)),
        "max_vector_norm": float(np.max(vector_norms)),
        "min_vector_norm": float(np.min(vector_norms)),
        "evaluation_sources": evaluation_sources,
        "top_k": TOP_K,
        "max_structural_hops": MAX_HOPS,
        "structural_true_positives": true_positive_count,
        "structural_false_positives": false_positive_count,
        "structural_precision": float(structural_precision),
        "evaluation_type": "project_defined_structural_proxy",
    }

    # t-SNE projection
    print("\nCreating t-SNE projection...")

    if len(labels) >= 3:
        perplexity = min(30, max(2, len(labels) - 1))

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
        )
        reduced = tsne.fit_transform(vectors)

        tsne_df = pd.DataFrame({
            "node": labels,
            "x": reduced[:, 0],
            "y": reduced[:, 1],
        })

        tsne_df.to_csv(
            os.path.join(
                results_dir,
                "embedding_tsne_projection.csv",
            ),
            index=False,
        )

        print("TSNE projection created:", reduced.shape)
        print("Saved: results/embedding_tsne_projection.csv")

    else:
        print("Too few embeddings for t-SNE projection.")
        tsne_df = pd.DataFrame({
            "node": labels,
            "x": np.nan,
            "y": np.nan,
        })

    # embedding diversity score
    if len(vectors) > 1:
        cosine_matrix = cosine_similarity(vectors)
        upper_triangle = cosine_matrix[
            np.triu_indices(len(cosine_matrix), k=1)
        ]
        diversity_score = 1 - np.mean(upper_triangle)
    else:
        diversity_score = 0.0

    print(
        "\nEmbedding Diversity Score:",
        round(float(diversity_score), 4),
    )

    embedding_stats["embedding_diversity_score"] = float(
        diversity_score
    )

    # save statistics
    statistics_output = os.path.join(
        results_dir,
        "embedding_statistics.json",
    )

    with open(statistics_output, "w", encoding="utf-8") as file:
        json.dump(
            embedding_stats,
            file,
            indent=4,
        )

    print("Saved:", statistics_output)
    print("\nEmbedding evaluation completed")

    return tsne_df


# main
if __name__ == "__main__":
    print(
        "This module is executed through main.py "
        "after Node2Vec training."
    )