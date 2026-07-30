from pyvis.network import Network
import json
import random
import networkx as nx
import os


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

def create_connected_sample(
    nodes,
    edges,
    sample_size=300
):

    print("\n--- CONNECTED KG SAMPLING ---")

    G = nx.DiGraph()

    for node in nodes:

        G.add_node(
            node["id"],
            **node
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

    UG = G.to_undirected()

    degrees = dict(UG.degree())

    important_nodes = sorted(
        degrees,
        key=degrees.get,
        reverse=True
    )

    start_node = important_nodes[0]

    sampled_nodes = set()

    queue = [start_node]

    while queue and len(sampled_nodes) < sample_size:

        current = queue.pop(0)

        if current not in sampled_nodes:

            sampled_nodes.add(current)

            neighbours = list(UG.neighbors(current))

            random.shuffle(neighbours)

            queue.extend(neighbours[:20])

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

    print("Sample nodes:", len(sampled_nodes_data))
    print("Sample edges:", len(sampled_edges))

    return sampled_nodes_data, sampled_edges


# ----------------------------------------------------
# NODE COLOURS
# ----------------------------------------------------

NODE_COLOURS = {

    "customer": "#4CAF50",
    "seller": "#2196F3",
    "order": "#FF9800",
    "product": "#9C27B0",
    "payment": "#F44336",
    "category": "#009688",
    "state": "#795548"

}


# ----------------------------------------------------
# BUILD VISUALIZATION
# ----------------------------------------------------

def build_visualization():

    data = load_graph_json()

    nodes = data["nodes"]
    edges = data["edges"]

    print("\n--- KG VISUALIZATION ---")

    print("Total nodes:", len(nodes))
    print("Total edges:", len(edges))

    sampled_nodes, sampled_edges = create_connected_sample(
        nodes,
        edges,
        sample_size=300
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        "results/sample_graph.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "nodes": sampled_nodes,
                "edges": sampled_edges
            },
            f,
            indent=4
        )

    net = Network(

        height="850px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="black"

    )

    # Better physics

    net.barnes_hut(
        gravity=-25000,
        central_gravity=0.2,
        spring_length=180,
        spring_strength=0.04
    )

    # ------------------------------------------------
    # Add Nodes
    # ------------------------------------------------

    for node in sampled_nodes:

        node_type = node.get(
            "type",
            "unknown"
        )

        colour = NODE_COLOURS.get(
            node_type,
            "#999999"
        )

        title = f"""
        <b>ID</b>: {node['id']}<br>
        <b>Type</b>: {node_type}<br>
        """

        if "city" in node:
            title += f"<b>City</b>: {node['city']}<br>"

        if "state" in node:
            title += f"<b>State</b>: {node['state']}<br>"

        if "status" in node:
            title += f"<b>Status</b>: {node['status']}<br>"

        if "payment_type" in node:
            title += f"<b>Payment</b>: {node['payment_type']}<br>"

        if "value" in node:
            title += f"<b>Value</b>: {node['value']}<br>"

        net.add_node(

            node["id"],

            label=node_type,

            title=title,

            color=colour,

            size=18

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
            ),

            arrows="to"

        )

    # ------------------------------------------------
    # Controls
    # ------------------------------------------------

    net.show_buttons(
        filter_=[
            "physics",
            "nodes",
            "edges"
        ]
    )

    # ------------------------------------------------
    # Interactive JavaScript
    # ------------------------------------------------

    net.set_options("""
    var options = {

      "interaction":{

        "hover":true,
        "navigationButtons":true,
        "keyboard":true,
        "multiselect":true

      },

      "physics":{

        "enabled":true

      }

    }
    """)

    html = net.generate_html()

    custom_js = """

<script>

network.on("doubleClick", function(params){

    if(params.nodes.length > 0){

        var node = params.nodes[0];

        var connected = network.getConnectedNodes(node);

        network.selectNodes(connected);

        network.fit({

            nodes: connected.concat([node]),

            animation:true

        });

    }

});

network.on("click", function(params){

    if(params.nodes.length>0){

        var node=params.nodes[0];

        console.log(node);

    }

});

</script>

"""

    html = html.replace(
        "</body>",
        custom_js + "</body>"
    )

    with open(
        "subgraph.html",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print(
        "Saved visualization → subgraph.html"
    )

    print(
        "Saved sampled graph → results/sample_graph.json"
    )


if __name__ == "__main__":

    build_visualization()