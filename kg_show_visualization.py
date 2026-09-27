import json
import os
import random

import networkx as nx
from pyvis.network import Network


# project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPH_JSON_PATH = os.path.join(RESULTS_DIR, "graph.json")
SAMPLE_GRAPH_PATH = os.path.join(RESULTS_DIR, "sample_graph.json")
HTML_PATH = os.path.join(BASE_DIR, "subgraph.html")


# visualization configuration
SAMPLE_SIZE = 2000
RANDOM_SEED = 42
MAX_NEIGHBOURS_PER_NODE = 50

NODE_COLOURS = {
    "customer": "#4CAF50",
    "seller": "#2196F3",
    "order": "#FF9800",
    "product": "#9C27B0",
    "payment": "#F44336",
    "category": "#009688",
    "review": "#607D8B",
    "sentiment": "#E91E63",
    "city": "#795548",
    "state": "#795548",
    "geographic_cluster": "#00ACC1",
}


# load exported KG
def load_graph_json():
    if not os.path.exists(GRAPH_JSON_PATH):
        raise FileNotFoundError(
            f"KG export not found: {GRAPH_JSON_PATH}"
        )

    with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# connected subgraph sampling
def create_connected_sample(
    nodes,
    edges,
    sample_size=SAMPLE_SIZE,
):
    print("\n--- CONNECTED KG SAMPLING ---")

    if not nodes:
        print("No nodes available for visualization.")
        return [], []

    random.seed(RANDOM_SEED)
    graph = nx.DiGraph()

    for node in nodes:
        graph.add_node(node["id"], **node)

    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            relation=edge.get("relation", ""),
            inferred=edge.get("inferred", False),
        )

    undirected = graph.to_undirected()
    degrees = dict(undirected.degree())

    start_node = max(degrees, key=degrees.get)

    sampled_nodes = set()
    queue = [start_node]

    while queue and len(sampled_nodes) < sample_size:
        current = queue.pop(0)

        if current in sampled_nodes:
            continue

        sampled_nodes.add(current)

        neighbours = list(undirected.neighbors(current))
        random.shuffle(neighbours)

        for neighbour in neighbours[:MAX_NEIGHBOURS_PER_NODE]:
            if neighbour not in sampled_nodes:
                queue.append(neighbour)

    sampled_edges = [
        edge
        for edge in edges
        if (
            edge["source"] in sampled_nodes
            and edge["target"] in sampled_nodes
        )
    ]

    sampled_nodes_data = [
        node
        for node in nodes
        if node["id"] in sampled_nodes
    ]

    print("Sample nodes:", len(sampled_nodes_data))
    print("Sample edges:", len(sampled_edges))

    return sampled_nodes_data, sampled_edges


# build node tooltip
def build_node_title(node):
    node_type = node.get("type", "unknown")

    title = (
        f"<b>ID</b>: {node.get('id', '')}<br>"
        f"<b>Type</b>: {node_type}<br>"
    )

    display_fields = [
        ("display_name", "Name"),
        ("city", "City"),
        ("state", "State"),
        ("status", "Status"),
        ("payment_type", "Payment"),
        ("value", "Value"),
        ("sentiment", "Sentiment"),
        ("nlp_sentiment", "NLP Sentiment"),
    ]

    for field, label in display_fields:
        if field in node:
            title += (
                f"<b>{label}</b>: "
                f"{node[field]}<br>"
            )

    if node.get("inferred"):
        title += (
            "<b>Knowledge status</b>: "
            "Inferred<br>"
        )

    return title


# build visualization
def build_visualization():
    data = load_graph_json()

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    print("\n--- KG VISUALIZATION ---")
    print("Total nodes:", len(nodes))
    print("Total edges:", len(edges))

    sampled_nodes, sampled_edges = create_connected_sample(
        nodes,
        edges,
        SAMPLE_SIZE,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(SAMPLE_GRAPH_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "nodes": sampled_nodes,
                "edges": sampled_edges,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    net = Network(
        height="850px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="black",
    )

    net.barnes_hut(
        gravity=-25000,
        central_gravity=0.2,
        spring_length=180,
        spring_strength=0.04,
    )

    # add nodes
    for node in sampled_nodes:
        node_type = node.get("type", "unknown")
        colour = NODE_COLOURS.get(node_type, "#999999")

        net.add_node(
            node["id"],
            label=node_type,
            title=build_node_title(node),
            color=colour,
            size=18,
        )

    # add edges
    for edge in sampled_edges:
        inferred = edge.get("inferred", False)

        edge_options = {
            "label": edge.get("relation", ""),
            "title": edge.get("relation", ""),
            "arrows": "to",
        }

        if inferred:
            edge_options.update({
                "dashes": True,
                "width": 2,
            })

            edge_options["title"] = (
                f"{edge.get('relation', '')}"
                "<br><b>Inferred by logical reasoning</b>"
            )

        net.add_edge(
            edge["source"],
            edge["target"],
            **edge_options,
        )

    # visualization controls
    net.show_buttons(
        filter_=[
            "physics",
            "nodes",
            "edges",
        ]
    )

    # interactive behaviour
    net.set_options(
        """
        var options = {
            "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true,
                "multiselect": true
            },
            "physics": {
                "enabled": true
            }
        }
        """
    )

    html = net.generate_html()

    # interactive JavaScript
    custom_js = """
    <script>
    network.on("doubleClick", function(params) {
        if (params.nodes.length > 0) {
            var node = params.nodes[0];
            var connected = network.getConnectedNodes(node);

            network.selectNodes(connected);

            network.fit({
                nodes: connected.concat([node]),
                animation: true
            });
        }
    });

    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var node = params.nodes[0];
            console.log("Selected KG node:", node);
        }
    });
    </script>
    """

    html = html.replace(
        "</body>",
        custom_js + "</body>",
    )

    with open(HTML_PATH, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"Saved visualization: {HTML_PATH}")
    print(f"Saved sampled graph: {SAMPLE_GRAPH_PATH}")


# main
if __name__ == "__main__":
    build_visualization()