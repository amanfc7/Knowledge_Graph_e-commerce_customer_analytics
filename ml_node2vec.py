from node2vec import Node2Vec
import numpy as np
import pickle
import os


# ----------------------------------------------------
# THEORY BRIDGE (REPORT / LO EXPLANATION)
# ----------------------------------------------------

def explain_gnn_relation():

    print("\n--- THEORY BRIDGE: KG ↔ GNN ↔ EMBEDDINGS ---")

    print("""
    1. Knowledge Graph:
       - Nodes represent entities/events
       - Edges represent semantic relationships

    2. GNN Message Passing:
       
       h_v^(k+1) = AGG(h_v^k, h_neighbors^k)

       Node representation is updated using neighbourhood information.

    3. Node2Vec:

       - Random walks explore graph neighbourhoods
       - Skip-gram learns node representations

       Similarity in embedding space represents structural similarity.

    4. Difference:

       Node2Vec:
       - fixed aggregation
       - no trainable message passing layers
       - unsupervised representation learning

       GNN:
       - learnable aggregation functions
       - task-specific training

    Therefore:
    Node2Vec provides a lightweight approximation of graph representation learning.
    """)



# ----------------------------------------------------
# NODE2VEC MODEL
# ----------------------------------------------------

def run_node2vec(G):

    print("\n--- KG EMBEDDINGS (NODE2VEC GRAPH LEARNING) ---")


    explain_gnn_relation()


    # ------------------------------------------------
    # Node2Vec Configuration
    # ------------------------------------------------

    node2vec = Node2Vec(
        G,
        dimensions=64,
        walk_length=20,
        num_walks=30,
        workers=2,
        p=1,
        q=1
    )


    print("\nTraining Node2Vec embedding model...")


    model = node2vec.fit(
        window=10,
        min_count=1,
        batch_words=128
    )


    print("Embedding training completed")


    # ------------------------------------------------
    # Save Model
    # ------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )


    model_path = "models/node2vec_model.pkl"


    with open(model_path, "wb") as f:
        pickle.dump(model, f)


    print(
        "Saved embedding model:",
        model_path
    )


    # ------------------------------------------------
    # Similarity Reasoning
    # ------------------------------------------------

    print(
        "\n--- LATENT KG REASONING USING EMBEDDINGS ---"
    )


    sample_nodes = list(G.nodes())[:5]


    for node in sample_nodes:


        node_string = str(node)


        if node_string in model.wv:


            print(
                "\nNode:",
                node
            )


            similar_nodes = model.wv.most_similar(
                node_string,
                topn=5
            )


            for similar, score in similar_nodes:

                print(
                    " ->",
                    similar,
                    " similarity:",
                    round(score,3)
                )


    return model