import json
import os



# ----------------------------------------------------
# KNOWLEDGE GRAPH EXPORTER
# JSON FORMAT FOR:
# - Visualization
# - GNN Processing
# - External KG Tools
# - Streamlit Application
# ----------------------------------------------------



def export_graph_to_json(
    G,
    filename="results/graph.json"
):


    print(
        "\n--- EXPORTING KG FOR DOWNSTREAM ANALYSIS ---"
    )



    # ==================================================
    # CREATE OUTPUT DIRECTORY
    # ==================================================


    folder = os.path.dirname(
        filename
    )


    if folder:


        os.makedirs(
            folder,
            exist_ok=True
        )



    # ==================================================
    # GRAPH METADATA
    # ==================================================


    metadata = {


        "graph_type":

        "Olist E-commerce Knowledge Graph",


        "nodes":

        G.number_of_nodes(),


        "edges":

        G.number_of_edges(),


        "directed":

        True

    }



    data = {


        "metadata":

        metadata,


        "nodes":[],

        "edges":[]

    }



    # ==================================================
    # EXPORT NODES
    # ==================================================


    for node, attr in G.nodes(data=True):


        node_data = {


            "id":

            str(node),


            "type":

            attr.get(

                "type",

                "unknown"

            ),



            "schema_type":

            attr.get(

                "schema_type",

                "unknown"

            ),



            "display_name":

            attr.get(

                "display_name",

                str(node)

            )

        }



        # ----------------------------------------------
        # Export all useful attributes
        # ----------------------------------------------


        ignored = [

            "type",

            "schema_type",

            "display_name"

        ]



        for key,value in attr.items():


            if key not in ignored:


                try:

                    node_data[key]=value


                except:

                    node_data[key]=str(value)



        data["nodes"].append(

            node_data

        )



    # ==================================================
    # EXPORT EDGES
    # ==================================================


    for source,target,attr in G.edges(data=True):


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



    # ==================================================
    # SAVE GRAPH JSON
    # ==================================================


    with open(

        filename,

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            data,

            file,

            indent=4,

            default=str

        )



    # ==================================================
    # SAVE METADATA FILE
    # ==================================================


    metadata_file = (

        os.path.dirname(filename)

        +

        "/graph_metadata.json"

    )


    with open(

        metadata_file,

        "w",

        encoding="utf-8"

    ) as file:


        json.dump(

            metadata,

            file,

            indent=4

        )



    # ==================================================
    # FILE INFORMATION
    # ==================================================


    file_size = (

        os.path.getsize(filename)

        /

        (1024*1024)

    )



    print(

        f"Graph exported → {filename}"

    )



    print(

        "Metadata exported →",

        metadata_file

    )



    print(

        "File size:",

        round(

            file_size,

            2

        ),

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