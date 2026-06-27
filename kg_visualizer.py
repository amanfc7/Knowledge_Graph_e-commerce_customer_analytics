import json

def export_graph_to_json(G, filename="graph.json"):

    data = {
        "nodes": [],
        "edges": []
    }

    for node, attr in G.nodes(data=True):
        data["nodes"].append({
            "id": node,
            "attributes": attr
        })

    for u, v, attr in G.edges(data=True):
        data["edges"].append({
            "source": u,
            "target": v,
            "relation": attr.get("relation", "")
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Graph exported to {filename}")