import networkx as nx


def run_graph_metrics(G):

    print("\n--- GRAPH STRUCTURAL ANALYSIS ---")

    print("Number of Nodes:", G.number_of_nodes())
    print("Number of Edges:", G.number_of_edges())

    # Density
    density = nx.density(G)

    print("\nGraph Density:")
    print(round(density, 6))


    # Degree centrality
    print("\n--- MOST CONNECTED KG NODES ---")

    degree = nx.degree_centrality(G)

    top_nodes = sorted(
        degree.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]


    for node, score in top_nodes:
        print(
            node,
            "centrality:",
            round(score,4),
            "type:",
            G.nodes[node].get("type")
        )


    # In-degree / Out-degree analysis

    print("\n--- RELATIONAL IMPORTANCE ---")

    in_degree = sorted(
        G.in_degree(),
        key=lambda x:x[1],
        reverse=True
    )[:10]


    for node, value in in_degree:

        print(
            "Node:",
            node,
            "incoming relations:",
            value,
            "type:",
            G.nodes[node].get("type")
        )