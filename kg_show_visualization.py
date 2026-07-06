<<<<<<< HEAD
from pyvis.network import Network
import json
import random

def load_graph_json():
    with open("graph.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_visualization():

    data = load_graph_json()

    nodes = data["nodes"]
    edges = data["edges"]

    print("Total nodes:", len(nodes))
    print("Total edges:", len(edges))

    # 🔥 STEP 1: SAMPLE NODES (NEVER empty graph again)
    sampled_nodes = random.sample(nodes, min(300, len(nodes)))
    node_ids = set(n["id"] for n in sampled_nodes)

    # 🔥 STEP 2: FILTER EDGES BASED ON SAMPLE
    sampled_edges = [
        e for e in edges
        if e["source"] in node_ids and e["target"] in node_ids
    ]

    print("Sample nodes:", len(sampled_nodes))
    print("Sample edges:", len(sampled_edges))

    # 🔥 STEP 3: BUILD GRAPH
    net = Network(height="750px", width="100%", directed=True, notebook=False)

    for n in sampled_nodes:
        net.add_node(n["id"], label=n.get("type", "node"))

    for e in sampled_edges:
        net.add_edge(e["source"], e["target"], label=e.get("relation", ""))

    net.write_html("subgraph.html")
    print("Saved: subgraph.html")


if __name__ == "__main__":
=======
from pyvis.network import Network
import json
import random
import networkx as nx

def load_graph_json():
    with open("graph.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_visualization():

    data = load_graph_json()

    nodes = data["nodes"]
    edges = data["edges"]

    print("Total nodes:", len(nodes))
    print("Total edges:", len(edges))

    # -----------------------------
    # STEP 1: BUILD TEMP NETWORKX GRAPH
    # -----------------------------
    G = nx.Graph()

    for n in nodes:
        G.add_node(n["id"])

    for e in edges:
        G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))

    # -----------------------------
    # STEP 2: TAKE A CONNECTED SUBGRAPH (IMPORTANT FIX)
    # -----------------------------
    largest_cc = max(nx.connected_components(G), key=len)
    sub_nodes = list(largest_cc)[:300]  # limit size safely

    subgraph = G.subgraph(sub_nodes)

    print("Sample nodes:", subgraph.number_of_nodes())
    print("Sample edges:", subgraph.number_of_edges())

    # -----------------------------
    # STEP 3: VISUALIZE
    # -----------------------------
    net = Network(height="750px", width="100%", directed=True)

    for node in subgraph.nodes():
        net.add_node(node, label=str(node))

    for u, v, attr in subgraph.edges(data=True):
        net.add_edge(u, v, label=attr.get("relation", ""))

    net.write_html("subgraph.html")
    print("Saved: subgraph.html")


if __name__ == "__main__":
>>>>>>> fa32367 (updated kg structure and analysis functions)
    build_visualization()