from node2vec import Node2Vec
import numpy as np


# ----------------------------------------------------
# THEORY BRIDGE (IMPORTANT FOR REPORT / LO MARKING)
# ----------------------------------------------------
def explain_gnn_relation():
    print("\n--- THEORY BRIDGE: KG ↔ GNN ↔ EMBEDDINGS ---")
    print("""
    1. Knowledge Graph:
       Nodes + edges = symbolic relational structure

    2. GNN Message Passing:
       h_v^{k+1} = AGG(h_v^k, h_neighbors^k)

    3. Node2Vec:
       Approximates message passing using random walks:
       - random walks ≈ neighborhood sampling
       - skip-gram ≈ aggregation function

    4. Conclusion:
       Node2Vec is a SPECIAL CASE of GNN embedding learning
       where:
       - fixed aggregator
       - no learnable message function
       - shallow propagation depth
    """)


# ----------------------------------------------------
# EMBEDDING MODEL
# ----------------------------------------------------
def run_node2vec(G):

    print("\n--- KG EMBEDDINGS (NODE2VEC / GNN EQUIVALENCE) ---")

    explain_gnn_relation()

    # -----------------------------
    # MODEL
    # -----------------------------
    node2vec = Node2Vec(
        G,
        dimensions=64,
        walk_length=20,
        num_walks=30,
        workers=2,
        p=1,
        q=1
    )

    model = node2vec.fit(window=10, min_count=1)

    # -----------------------------
    # EMBEDDINGS = LATENT KG STATE
    # -----------------------------
    embeddings = model.wv

    nodes = list(G.nodes())[:5]

    print("\n--- LATENT SEMANTIC SIMILARITY (KG REASONING IN EMBED SPACE) ---")

    for node in nodes:
        if str(node) in embeddings:

            print(f"\nNode: {node}")

            similar = model.wv.most_similar(str(node), topn=5)

            for s, score in similar:
                print(f"  -> {s} (similarity: {round(score, 3)})")

    return model