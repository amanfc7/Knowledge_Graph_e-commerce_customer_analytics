import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import json



# ----------------------------------------------------
# EMBEDDING QUALITY EVALUATION
# KG REPRESENTATION LEARNING ANALYSIS
# ----------------------------------------------------


def evaluate_embeddings(model):


    print("\n--- EMBEDDING QUALITY ANALYSIS ---")


    # ------------------------------------------------
    # Create results directory
    # ------------------------------------------------


    os.makedirs(
        "results",
        exist_ok=True
    )



    vectors = []

    labels = []



    # ------------------------------------------------
    # Sample embeddings
    # ------------------------------------------------


    for node in model.wv.index_to_key[:300]:


        vectors.append(
            model.wv[node]
        )

        labels.append(
            node
        )



    vectors = np.array(
        vectors
    )



    print(
        "Embedding dimension:",
        vectors.shape[1]
    )


    print(
        "Number of sampled nodes:",
        len(labels)
    )



    # ------------------------------------------------
    # Similarity Evaluation
    # ------------------------------------------------


    print(
        "\n--- NODE SIMILARITY ANALYSIS ---"
    )


    similarity_results = []



    for node in labels[:10]:


        similar_nodes = model.wv.most_similar(
            node,
            topn=5
        )


        for neighbour, score in similar_nodes:


            similarity_results.append(

                {

                    "source_node":
                    node,


                    "similar_node":
                    neighbour,


                    "similarity_score":
                    round(
                        float(score),
                        4
                    )

                }

            )



            print(
                node,
                "->",
                neighbour,
                ":",
                round(score,4)
            )



    similarity_df = pd.DataFrame(
        similarity_results
    )



    similarity_df.to_csv(

        "results/node_similarity_results.csv",

        index=False

    )



    print(
        "Saved:",
        "results/node_similarity_results.csv"
    )



    # ------------------------------------------------
    # Vector Statistics
    # ------------------------------------------------


    vector_norms = np.linalg.norm(
        vectors,
        axis=1
    )


    embedding_stats = {


        "embedding_dimension":
        int(vectors.shape[1]),


        "sample_nodes":
        len(labels),


        "average_vector_norm":
        float(
            np.mean(vector_norms)
        ),


        "max_vector_norm":
        float(
            np.max(vector_norms)
        ),


        "min_vector_norm":
        float(
            np.min(vector_norms)
        )

    }



    with open(

        "results/embedding_statistics.json",

        "w"

    ) as file:


        json.dump(

            embedding_stats,

            file,

            indent=4

        )



    print(
        "Saved:",
        "results/embedding_statistics.json"
    )



    # ------------------------------------------------
    # t-SNE Projection
    # ------------------------------------------------


    print(
        "\nCreating t-SNE projection..."
    )



    tsne = TSNE(

        n_components=2,

        random_state=42,

        perplexity=30

    )



    reduced = tsne.fit_transform(
        vectors
    )



    print(
        "TSNE projection created:",
        reduced.shape
    )



    tsne_df = pd.DataFrame(

        {

            "node":
            labels,


            "x":
            reduced[:,0],


            "y":
            reduced[:,1]

        }

    )



    tsne_df.to_csv(

        "results/embedding_tsne_projection.csv",

        index=False

    )



    print(
        "Saved:",
        "results/embedding_tsne_projection.csv"
    )



    # ------------------------------------------------
    # Embedding Diversity Score
    # ------------------------------------------------


    cosine_matrix = cosine_similarity(
        vectors
    )


    upper_triangle = cosine_matrix[
        np.triu_indices(
            len(cosine_matrix),
            k=1
        )
    ]



    diversity_score = 1 - np.mean(
        upper_triangle
    )



    print(
        "\nEmbedding Diversity Score:",
        round(
            diversity_score,
            4
        )
    )



    embedding_stats["embedding_diversity_score"] = float(
        diversity_score
    )



    with open(

        "results/embedding_statistics.json",

        "w"

    ) as file:


        json.dump(

            embedding_stats,

            file,

            indent=4

        )



    print(
        "\nEmbedding evaluation completed"
    )



    return tsne_df