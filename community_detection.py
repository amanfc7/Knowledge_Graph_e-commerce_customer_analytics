import community.community_louvain as community_louvain
import networkx as nx


def run_community_detection(G):

    print("\n--- COMMUNITY DETECTION ---")

    # convert directed graph
    UG = G.to_undirected()

    partition = community_louvain.best_partition(
        UG,
        random_state=42
    )

    communities = {}

    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)


    print("Total Communities:", len(communities))


    sizes = sorted(
        [(c,len(nodes)) for c,nodes in communities.items()],
        key=lambda x:x[1],
        reverse=True
    )


    print("\nLargest Communities")

    for c,size in sizes[:10]:
        print(
            "Community:",
            c,
            "Size:",
            size
        )


    nx.set_node_attributes(
        G,
        partition,
        "community"
    )

    return partition