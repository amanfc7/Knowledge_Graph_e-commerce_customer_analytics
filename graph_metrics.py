import os
import pandas as pd
import networkx as nx

# configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
REASONING_DIR = os.path.join(RESULTS_DIR, "reasoning")


def save_validation_result(name, dataframe):
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    path = os.path.join(VALIDATION_DIR, f"{name}.csv")
    dataframe.to_csv(path, index=False)
    print(f"Validation result saved to: {path}")


def analyse_node_types(G):
    counts = {}

    for _, attributes in G.nodes(data=True):
        node_type = attributes.get("type", "unknown")
        counts[node_type] = counts.get(node_type, 0) + 1

    dataframe = pd.DataFrame(
        [
            {
                "node_type": node_type,
                "node_count": count,
            }
            for node_type, count in sorted(counts.items())
        ]
    )

    print("\n--- NODE TYPE DISTRIBUTION ---")
    print(dataframe.to_string(index=False))
    save_validation_result("node_type_counts", dataframe)

    return dataframe


def analyse_relationship_types(G):
    counts = {}

    for _, _, attributes in G.edges(data=True):
        relation = attributes.get("relation", "unknown")
        counts[relation] = counts.get(relation, 0) + 1

    dataframe = pd.DataFrame(
        [
            {
                "relationship_type": relation,
                "edge_count": count,
            }
            for relation, count in sorted(counts.items())
        ]
    )

    print("\n--- RELATIONSHIP TYPE DISTRIBUTION ---")
    print(dataframe.to_string(index=False))
    save_validation_result(
        "relationship_type_counts",
        dataframe,
    )

    return dataframe


def analyse_explicit_inferred_edges(G):
    explicit_edges = 0
    inferred_edges = 0

    for _, _, attributes in G.edges(data=True):
        if (
            attributes.get("inferred", False)
            or attributes.get("inference_rule")
            or attributes.get("source")
            == "logical_forward_chaining"
        ):
            inferred_edges += 1
        else:
            explicit_edges += 1

    dataframe = pd.DataFrame(
        [
            {
                "edge_type": "explicit",
                "edge_count": explicit_edges,
            },
            {
                "edge_type": "inferred",
                "edge_count": inferred_edges,
            },
        ]
    )

    print("\n--- EXPLICIT VS INFERRED EDGES ---")
    print(dataframe.to_string(index=False))

    save_validation_result(
        "explicit_vs_inferred_edges",
        dataframe,
    )

    return dataframe


def calculate_reasoning_expansion(G):
    reasoning_summary_path = os.path.join(
        REASONING_DIR,
        "reasoning_summary.csv",
    )

    if not os.path.exists(reasoning_summary_path):
        print("\n--- REASONING EXPANSION ---")
        print("Reasoning summary not found.")
        return None

    try:
        summary = pd.read_csv(reasoning_summary_path)

        metrics = dict(
            zip(
                summary["metric"],
                summary["value"],
            )
        )

        explicit_facts = int(
            metrics["explicit_facts"]
        )

        inferred_facts = int(
            metrics["new_inferred_facts"]
        )

        final_facts = int(
            metrics["total_facts_after_reasoning"]
        )

        if explicit_facts > 0:
            expansion_ratio = (
                final_facts / explicit_facts
            )

            edge_increase_percent = (
                inferred_facts / explicit_facts
            ) * 100
        else:
            expansion_ratio = 0
            edge_increase_percent = 0

        dataframe = pd.DataFrame(
            [
                {
                    "explicit_edges_before_reasoning":
                        explicit_facts,
                    "inferred_edges_added":
                        inferred_facts,
                    "final_edges_after_reasoning":
                        final_facts,
                    "expansion_ratio":
                        expansion_ratio,
                    "edge_increase_percent":
                        edge_increase_percent,
                }
            ]
        )

        print("\n--- REASONING EXPANSION ---")
        print(
            "Explicit edges before reasoning :",
            f"{explicit_facts:,}",
        )
        print(
            "Inferred edges added             :",
            f"{inferred_facts:,}",
        )
        print(
            "Final edges after reasoning      :",
            f"{final_facts:,}",
        )
        print(
            "Reasoning expansion ratio        :",
            f"{expansion_ratio:.2f}x",
        )
        print(
            "Edge increase                    :",
            f"{edge_increase_percent:.2f}%",
        )

        save_validation_result(
            "reasoning_expansion",
            dataframe,
        )

        return dataframe

    except Exception as error:
        print("\n--- REASONING EXPANSION ---")
        print(
            "Reasoning expansion could not be calculated:",
            error,
        )
        return None


def validate_graph_integrity(G):
    missing_node_types = sum(
        1
        for _, attributes in G.nodes(data=True)
        if not attributes.get("type")
    )

    missing_relationship_types = sum(
        1
        for _, _, attributes in G.edges(data=True)
        if not attributes.get("relation")
    )

    validation = pd.DataFrame(
        [
            {
                "check": "nodes_without_type",
                "count": missing_node_types,
                "status": (
                    "PASS"
                    if missing_node_types == 0
                    else "CHECK"
                ),
            },
            {
                "check": "edges_without_relation",
                "count": missing_relationship_types,
                "status": (
                    "PASS"
                    if missing_relationship_types == 0
                    else "CHECK"
                ),
            },
        ]
    )

    print("\n--- KG STRUCTURAL INTEGRITY VALIDATION ---")
    print(validation.to_string(index=False))

    save_validation_result(
        "graph_integrity",
        validation,
    )

    return validation


def run_graph_metrics(G):
    print("\n==============================")
    print(" GRAPH STRUCTURAL ANALYSIS")
    print("==============================")

    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    print("Nodes               :", f"{nodes:,}")
    print("Edges               :", f"{edges:,}")

    density = nx.density(G)
    print("Density             :", f"{density:.2e}")

    total_degree = sum(
        dict(G.degree()).values()
    )

    average_degree = (
        total_degree / nodes
        if nodes > 0
        else 0
    )

    print(
        "Average Degree      :",
        round(average_degree, 2),
    )

    if nodes > 0:
        UG = G.to_undirected()

        component_count = (
            nx.number_connected_components(UG)
        )

        largest_component_size = max(
            (
                len(component)
                for component
                in nx.connected_components(UG)
            ),
            default=0,
        )
    else:
        component_count = 0
        largest_component_size = 0

    print(
        "Connected Components:",
        component_count,
    )

    print(
        "Largest Component   :",
        largest_component_size,
    )

    # explicit vs inferred edges
    explicit_inferred = (
        analyse_explicit_inferred_edges(G)
    )

    # reasoning expansion
    reasoning_expansion = (
        calculate_reasoning_expansion(G)
    )

    # most connected entities
    print("\n--- MOST CONNECTED ENTITIES ---")

    degrees = dict(G.degree())

    top_degree = sorted(
        degrees.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    for node, degree in top_degree:
        print(
            "Node:",
            node,
            "| Degree:",
            degree,
            "| Type:",
            G.nodes[node].get("type"),
        )

    # LO7 — KG construction validation
    print(
        "\n--- LO7 KNOWLEDGE GRAPH "
        "CONSTRUCTION VALIDATION ---"
    )

    analyse_node_types(G)
    analyse_relationship_types(G)
    validate_graph_integrity(G)

    # overall structural summary
    summary_rows = [
        {
            "metric": "nodes",
            "value": nodes,
        },
        {
            "metric": "edges",
            "value": edges,
        },
        {
            "metric": "density",
            "value": density,
        },
        {
            "metric": "average_degree",
            "value": average_degree,
        },
        {
            "metric": "connected_components",
            "value": component_count,
        },
        {
            "metric": "largest_component",
            "value": largest_component_size,
        },
    ]

    if reasoning_expansion is not None:
        expansion = reasoning_expansion.iloc[0]

        summary_rows.extend(
            [
                {
                    "metric": "explicit_edges",
                    "value": expansion[
                        "explicit_edges_before_reasoning"
                    ],
                },
                {
                    "metric": "inferred_edges",
                    "value": expansion[
                        "inferred_edges_added"
                    ],
                },
                {
                    "metric": "reasoning_expansion_ratio",
                    "value": expansion[
                        "expansion_ratio"
                    ],
                },
                {
                    "metric":
                        "reasoning_edge_increase_percent",
                    "value": expansion[
                        "edge_increase_percent"
                    ],
                },
            ]
        )

    summary = pd.DataFrame(summary_rows)

    save_validation_result(
        "graph_summary",
        summary,
    )

    print("\nGRAPH ANALYSIS COMPLETED")
    print(
        "LO7 KG CONSTRUCTION VALIDATION COMPLETED"
    )