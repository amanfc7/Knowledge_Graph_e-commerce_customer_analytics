import numpy as np
from sklearn.manifold import TSNE


def evaluate_embeddings(model):

    print("\n--- EMBEDDING QUALITY ANALYSIS ---")

    vectors = []
    labels = []

    for node in model.wv.index_to_key[:300]:
        vectors.append(model.wv[node])
        labels.append(node)

    vectors = np.array(vectors)

    print("Embedding dimension:", vectors.shape[1])
    print("Number of sampled nodes:", len(labels))

    # Similarity example
    node = labels[0]

    print("\nExample similarity:")
    for neighbour, score in model.wv.most_similar(node, topn=5):
        print(f"{neighbour} : {score:.4f}")

    # t-SNE projection
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )

    reduced = tsne.fit_transform(vectors)

    print("\nTSNE projection created:", reduced.shape)

    print("Embedding evaluation completed")