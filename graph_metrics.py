import networkx as nx


def run_graph_metrics(G):

    print("\n==============================")
    print(" GRAPH STRUCTURAL ANALYSIS")
    print("==============================")


    # ----------------------------------------------------
    # BASIC GRAPH STATISTICS
    # ----------------------------------------------------

    nodes = G.number_of_nodes()

    edges = G.number_of_edges()


    print(
        "Nodes               :",
        f"{nodes:,}"
    )


    print(
        "Edges               :",
        f"{edges:,}"
    )



    # ----------------------------------------------------
    # GRAPH DENSITY
    # ----------------------------------------------------

    density = nx.density(G)


    print(
        "Density             :",
        round(
            density,
            8
        )
    )



    # ----------------------------------------------------
    # AVERAGE DEGREE
    # ----------------------------------------------------

    degrees = dict(
        G.degree()
    )


    average_degree = (

        sum(degrees.values())
        /
        nodes

    )


    print(
        "Average Degree      :",
        round(
            average_degree,
            2
        )
    )



    # ----------------------------------------------------
    # CONNECTED COMPONENT ANALYSIS
    # ----------------------------------------------------

    UG = G.to_undirected()


    components = list(
        nx.connected_components(UG)
    )


    largest_component = max(
        components,
        key=len
    )


    print(
        "Connected Components:",
        len(components)
    )


    print(
        "Largest Component   :",
        len(largest_component)
    )



    # ----------------------------------------------------
    # CLUSTERING COEFFICIENT
    # ----------------------------------------------------

    clustering = nx.average_clustering(
        UG
    )


    print(
        "Average Clustering  :",
        round(
            clustering,
            4
        )
    )



    # ----------------------------------------------------
    # DEGREE CENTRALITY
    # ----------------------------------------------------

    print(
        "\n--- MOST CONNECTED KG NODES ---"
    )


    degree_centrality = nx.degree_centrality(
        G
    )


    top_nodes = sorted(

        degree_centrality.items(),

        key=lambda x:x[1],

        reverse=True

    )[:10]



    for node, score in top_nodes:


        print(

            "Node:",
            node,

            "| type:",
            G.nodes[node].get(
                "type"
            ),

            "| centrality:",
            round(
                score,
                5
            )

        )



    # ----------------------------------------------------
    # RELATIONAL IMPORTANCE
    # ----------------------------------------------------

    print(
        "\n--- MOST CONNECTED ENTITIES ---"
    )


    top_degree = sorted(

        degrees.items(),

        key=lambda x:x[1],

        reverse=True

    )[:10]



    for node, degree in top_degree:


        print(

            "Node:",
            node,

            "| Degree:",
            degree,

            "| Type:",
            G.nodes[node].get(
                "type"
            )

        )



    print(
        "\nGRAPH ANALYSIS COMPLETED"
    )