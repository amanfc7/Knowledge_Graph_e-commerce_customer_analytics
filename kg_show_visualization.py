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
    build_visualization()