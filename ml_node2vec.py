from node2vec import Node2Vec

def run_node2vec(G):

    print("\n--- NODE2VEC EMBEDDINGS (EXCEED LO) ---")

    node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=50, workers=2)

    model = node2vec.fit()

    # Example similarity
    nodes = list(G.nodes())[:5]

    for node in nodes:
        try:
            print(f"\nSimilar to {node}:")
            print(model.wv.most_similar(str(node), topn=5))
        except:
            pass

    return model