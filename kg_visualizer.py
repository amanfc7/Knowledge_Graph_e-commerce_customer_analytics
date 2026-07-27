import json
import os



# ----------------------------------------------------
# KNOWLEDGE GRAPH EXPORTER
# JSON FORMAT FOR:
# - Visualization
# - GNN Processing
# - External KG Tools
# ----------------------------------------------------

def export_graph_to_json(G, filename="graph.json"):


    print("\n--- EXPORTING KG FOR DOWNSTREAM ANALYSIS ---")



    data = {

        "metadata": {

            "graph_type": "E-commerce Knowledge Graph",

            "nodes": G.number_of_nodes(),

            "edges": G.number_of_edges(),

            "directed": True

        },


        "nodes": [],

        "edges": []

    }



    # ------------------------------------------------
    # Export Nodes
    # ------------------------------------------------

    for node, attr in G.nodes(data=True):


        node_data = {


            "id": str(node),


            "type":
                attr.get(
                    "type",
                    "unknown"
                ),


            "schema_type":
                attr.get(
                    "schema_type",
                    "unknown"
                )

        }


        # Store numerical attributes
        # Example:
        # payment value

        if "value" in attr:

            node_data["value"] = attr["value"]



        data["nodes"].append(
            node_data
        )



    # ------------------------------------------------
    # Export Edges
    # ------------------------------------------------

    for source, target, attr in G.edges(data=True):


        edge_data = {


            "source":
                str(source),


            "target":
                str(target),


            "relation":
                attr.get(
                    "relation",
                    "CONNECTED"
                ),


            "weight":
                attr.get(
                    "weight",
                    1.0
                )

        }



        data["edges"].append(
            edge_data
        )



    # ------------------------------------------------
    # Save JSON
    # ------------------------------------------------


    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(
            data,
            file,
            indent=4
        )



    file_size = os.path.getsize(filename) / (1024 * 1024)


    print(
        f"Graph exported → {filename}"
    )


    print(
        "File size:",
        round(file_size,2),
        "MB"
    )


    print(
        "Exported nodes:",
        len(data["nodes"])
    )


    print(
        "Exported edges:",
        len(data["edges"])
    )


    return data