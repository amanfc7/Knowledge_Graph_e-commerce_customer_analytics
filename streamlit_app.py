import streamlit as st
import json
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import tempfile


# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Olist Knowledge Graph Explorer",
    layout="wide"
)


# =====================================================
# LOAD GRAPH JSON
# =====================================================


@st.cache_data(show_spinner="Loading KG JSON...")
def load_graph():

    with open(
        "results/graph.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



# =====================================================
# CREATE NETWORKX GRAPH
# =====================================================


@st.cache_resource(show_spinner="Building NetworkX KG...")
def create_networkx_graph(data):

    G = nx.DiGraph()


    for node in data["nodes"]:

        G.add_node(

            node["id"],

            type=node.get(
                "type",
                "unknown"
            ),

            schema_type=node.get(
                "schema_type",
                ""
            )

        )


    for edge in data["edges"]:

        G.add_edge(

            edge["source"],

            edge["target"],

            relation=edge.get(
                "relation",
                ""
            )

        )


    return G



# =====================================================
# LOAD CSV RESULTS
# =====================================================


@st.cache_data
def load_results():

    results={}


    if os.path.exists("results"):


        for file in os.listdir("results"):


            if file.endswith(".csv"):


                results[file]=pd.read_csv(
                    f"results/{file}"
                )


    return results



# =====================================================
# PYVIS LOCAL VIEW
# =====================================================


def create_pyvis_graph(
        G,
        selected_node
):


    net = Network(

        height="700px",

        width="100%",

        directed=True,

        notebook=False

    )


    nodes=set()


    nodes.add(selected_node)



    # limit neighbours

    neighbours=list(
        G.successors(selected_node)
    )[:50]


    neighbours += list(
        G.predecessors(selected_node)
    )[:50]


    nodes.update(neighbours)



    H=G.subgraph(nodes)



    for node,data in H.nodes(data=True):


        net.add_node(

            node,

            label=str(node)[:25],

            title=f"""
Node:
{node}

Type:
{data.get('type')}

Schema:
{data.get('schema_type')}
""",

        )



    for u,v,data in H.edges(data=True):


        net.add_edge(

            u,

            v,

            label=data.get(
                "relation",
                ""
            )

        )


    net.repulsion(

        node_distance=180,

        central_gravity=0.2,

        spring_length=120

    )


    path="temp_graph.html"


    net.save_graph(path)


    return path



# =====================================================
# MAIN
# =====================================================


st.title(
    "🧠 Olist Knowledge Graph Explorer"
)


st.write(
"""
Large Scale E-Commerce Knowledge Graph

Entities:

- Customers
- Orders
- Products
- Sellers
- Payments
- Categories
- Reviews
- Geography
"""
)



# Load

data=load_graph()

G=create_networkx_graph(data)

results=load_results()



# =====================================================
# SIDEBAR
# =====================================================


mode=st.sidebar.selectbox(

    "Choose Module",

    [

        "Graph Explorer",

        "Analytics",

        "Search Entity",

        "Graph Metrics"

    ]

)



# =====================================================
# GRAPH EXPLORER
# =====================================================


if mode=="Graph Explorer":


    st.header(
        "Interactive KG Explorer"
    )


    col1,col2,col3=st.columns(3)


    col1.metric(
        "Nodes",
        f"{G.number_of_nodes():,}"
    )


    col2.metric(
        "Edges",
        f"{G.number_of_edges():,}"
    )


    col3.metric(
        "Node Types",
        len(
            set(
                nx.get_node_attributes(
                    G,
                    "type"
                ).values()
            )
        )
    )



    query=st.text_input(

        "Search entity"

    )



    if query:


        matches=[

            n for n in G.nodes()

            if query.lower()
            in str(n).lower()

        ][:50]



    else:


        matches=list(
            G.nodes()
        )[:50]



    selected=st.selectbox(

        "Select Node",

        matches

    )



    html=create_pyvis_graph(

        G,

        selected

    )


    with open(
        html,
        "r",
        encoding="utf-8"
    ) as f:

        graph_html=f.read()



    st.components.v1.html(

        graph_html,

        height=750

    )



    st.subheader(
        "Node Information"
    )


    st.json(

        G.nodes[selected]

    )




    st.subheader(
        "Relationships"
    )


    relations=[]



    for n in G.successors(selected):


        relations.append(

            {

            "direction":"OUT",

            "node":n,

            "relation":
            G[selected][n]
            .get(
                "relation"
            )

            }

        )



    for n in G.predecessors(selected):


        relations.append(

            {

            "direction":"IN",

            "node":n,

            "relation":
            G[n][selected]
            .get(
                "relation"
            )

            }

        )



    st.dataframe(

        pd.DataFrame(
            relations
        )

    )



# =====================================================
# ANALYTICS
# =====================================================


elif mode=="Analytics":


    st.header(
        "KG Business Analytics"
    )


    file=st.selectbox(

        "Select Result",

        list(results.keys())

    )


    st.dataframe(

        results[file]

    )




# =====================================================
# SEARCH
# =====================================================


elif mode=="Search Entity":


    st.header(
        "Entity Search"
    )


    query=st.text_input(
        "Search"
    )



    if query:


        results=[

            n for n in G.nodes()

            if query.lower()
            in str(n).lower()

        ]


        st.write(
            len(results),
            "matches"
        )


        for r in results[:100]:

            st.write(

                r,

                G.nodes[r]

            )




# =====================================================
# METRICS
# =====================================================


elif mode=="Graph Metrics":


    st.header(
        "Structural Metrics"
    )


    st.metric(

        "Nodes",

        G.number_of_nodes()

    )


    st.metric(

        "Edges",

        G.number_of_edges()

    )


    st.metric(

        "Density",

        round(
            nx.density(G),
            8
        )

    )


    st.metric(

        "Connected Components",

        nx.number_weakly_connected_components(G)

    )


    st.success(
        "Knowledge Graph loaded successfully"
    )