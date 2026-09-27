
import json
import os


# knowledge graph exporter
# exports the NetworkX KG for visualization, Streamlit,
# external tools and downstream graph analysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def export_graph_to_json(G, filename=None):
    print(
        "\n--- EXPORTING KG FOR DOWNSTREAM ANALYSIS ---"
    )

    # output path
    if filename is None:
        filename = os.path.join(
            RESULTS_DIR,
            "graph.json",
        )

    folder = os.path.dirname(filename)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True,
        )

    # graph metadata
    inferred_edges = sum(
        1
        for _, _, attr in G.edges(data=True)
        if attr.get("inferred", False)
    )

    metadata = {
        "graph_type": (
            "Olist E-commerce Knowledge Graph"
        ),
        "representation": (
            "Property-graph-style NetworkX graph"
        ),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "explicit_edges": (
            G.number_of_edges() - inferred_edges
        ),
        "inferred_edges": inferred_edges,
        "directed": True,
        "source": (
            "Olist Brazilian E-Commerce Dataset"
        ),
        "export_format": "JSON",
        "purpose": (
            "Visualization, Streamlit service "
            "and downstream graph analysis"
        ),
    }

    data = {
        "metadata": metadata,
        "nodes": [],
        "edges": [],
    }

    # export nodes
    for node, attr in G.nodes(data=True):
        node_data = {
            "id": str(node),
            "type": attr.get(
                "type",
                "unknown",
            ),
            "schema_type": attr.get(
                "schema_type",
                "unknown",
            ),
            "display_name": attr.get(
                "display_name",
                str(node),
            ),
        }

        ignored = {
            "type",
            "schema_type",
            "display_name",
        }

        for key, value in attr.items():
            if key not in ignored:
                node_data[key] = value

        data["nodes"].append(node_data)

    # export edges
    for source, target, attr in G.edges(data=True):
        edge_data = {
            "source": str(source),
            "target": str(target),
            "relation": attr.get(
                "relation",
                "CONNECTED",
            ),
            "weight": attr.get(
                "weight",
                1.0,
            ),
            "inferred": attr.get(
                "inferred",
                False,
            ),
        }

        # preserve logical reasoning provenance
        if attr.get("inferred", False):
            edge_data["inference_rule"] = attr.get(
                "inference_rule"
            )
            edge_data["inference_iteration"] = attr.get(
                "inference_iteration"
            )
            edge_data["knowledge_source"] = attr.get(
                "source",
                "logical_forward_chaining",
            )
        else:
            edge_data["knowledge_source"] = attr.get(
                "source",
                "source_dataset",
            )

        data["edges"].append(edge_data)

    # save graph JSON
    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # save metadata
    metadata_file = os.path.join(
        folder,
        "graph_metadata.json",
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # file information
    file_size = (
        os.path.getsize(filename)
        / (1024 * 1024)
    )

    print(
        f"Graph exported: {filename}"
    )
    print(
        f"Metadata exported: {metadata_file}"
    )
    print(
        f"File size: {file_size:.2f} MB"
    )
    print(
        f"Exported nodes: {len(data['nodes']):,}"
    )
    print(
        f"Exported edges: {len(data['edges']):,}"
    )
    print(
        f"Explicit edges: {metadata['explicit_edges']:,}"
    )
    print(
        f"Inferred edges: {metadata['inferred_edges']:,}"
    )

    return data
