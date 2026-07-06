import json


def export_graph_to_json(G, filename="graph.json"):

    print("\n--- EXPORTING KG FOR DOWNSTREAM ML/GNN ---")

    data = {
        "nodes": [],
        "edges": []
    }

    # -------------------------
    # NODES
    # -------------------------
    for node, attr in G.nodes(data=True):
        data["nodes"].append({
            "id": node,
            "type": attr.get("type", "unknown"),
            "schema_type": attr.get("schema_type", "unknown")
        })

    # -------------------------
    # EDGES
    # -------------------------
    for u, v, attr in G.edges(data=True):
        data["edges"].append({
            "source": u,
            "target": v,
            "relation": attr.get("relation", "linked"),
            "weight": attr.get("weight", 1.0)
        })

    # -------------------------
    # SAVE
    # -------------------------
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Graph exported → {filename}")