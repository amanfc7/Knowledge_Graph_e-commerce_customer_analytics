from pyvis.network import Network
import json
import random
import networkx as nx



# ----------------------------------------------------
# LOAD EXPORTED KG
# ----------------------------------------------------

def load_graph_json():

    with open(
        "graph.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)



# ----------------------------------------------------
# CONNECTED SUBGRAPH SAMPLING
# ----------------------------------------------------

def create_connected_sample(nodes, edges, sample_size=300):


    print("\n--- CONNECTED KG SAMPLING ---")


    # Create NetworkX graph from JSON

    G = nx.DiGraph()

    for node in nodes:

        G.add_node(
            node["id"],
            type=node.get(
                "type",
                "unknown"
            )
        )

    for edge in edges:

        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge.get(
                "relation",
                ""
            )
        )


    # Convert to undirected graph for connected traversal

    UG = G.to_undirected()


    # ------------------------------------------------
    # Select important starting nodes
    # ------------------------------------------------

    degrees = dict(
        UG.degree()
    )


    important_nodes = sorted(
        degrees,
        key=degrees.get,
        reverse=True
    )


    # Pick high-connected starting node

    start_node = important_nodes[0]


    sampled_nodes = set()


    # BFS neighbourhood expansion

    queue = [start_node]


    while queue and len(sampled_nodes) < sample_size:


        current = queue.pop(0)


        if current not in sampled_nodes:

            sampled_nodes.add(
                current
            )


            # Undirected neighbours

            neighbours = list(
                UG.neighbors(current)
            )


            random.shuffle(
                neighbours
            )


            queue.extend(
                neighbours[:20]
            )



    sampled_edges = [

        e for e in edges

        if e["source"] in sampled_nodes

        and

        e["target"] in sampled_nodes

    ]



    sampled_nodes_data = [

        n for n in nodes

        if n["id"] in sampled_nodes

    ]



    print(
        "Sample nodes:",
        len(sampled_nodes_data)
    )


    print(
        "Sample edges:",
        len(sampled_edges)
    )


    return (
        sampled_nodes_data,
        sampled_edges
    )



# ----------------------------------------------------
# BUILD VISUALIZATION
# ----------------------------------------------------

def build_visualization():


    data = load_graph_json()


    nodes = data["nodes"]

    edges = data["edges"]



    print("\n--- KG VISUALIZATION ---")


    print(
        "Total nodes:",
        len(nodes)
    )


    print(
        "Total edges:",
        len(edges)
    )



    sampled_nodes, sampled_edges = create_connected_sample(
        nodes,
        edges,
        sample_size=300
    )


    # ------------------------------------------------
    # PyVis Network
    # ------------------------------------------------


    net = Network(

        height="800px",

        width="100%",

        directed=True,

        notebook=False

    )


    # ------------------------------------------------
    # Node Colours
    # ------------------------------------------------

    colors = {

        "customer": "#3498db",

        "order": "#2ecc71",

        "product": "#e74c3c",

        "seller": "#f39c12",

        "payment": "#9b59b6",

        "category": "#1abc9c",

        "state": "#34495e",

        "unknown": "#95a5a6"
    }


    # ------------------------------------------------
    # Add Nodes
    # ------------------------------------------------


    for node in sampled_nodes:


        net.add_node(

            node["id"],

            label=str(node["id"])[:8],

            color=colors.get(
                node["type"],
                "#95a5a6"
            ),

            title=(
                "Entity: "
                +
                str(node["id"])
                +
                "\nType: "
                +
                node["type"]
            )

        )



    # ------------------------------------------------
    # Add Edges
    # ------------------------------------------------


    for edge in sampled_edges:


        net.add_edge(

            edge["source"],

            edge["target"],

            label=edge.get(
                "relation",
                ""
            ),

            title=edge.get(
                "relation",
                ""
            )

        )



    # ------------------------------------------------
    # Save HTML
    # ------------------------------------------------


    net.write_html(
        "subgraph.html"
    )


    print(
        "Saved visualization → subgraph.html"
    )



if __name__ == "__main__":

    build_visualization()