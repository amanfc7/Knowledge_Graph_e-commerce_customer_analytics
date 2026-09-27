import json
import os
import random

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network

from graph_queries import ask_knowledge_graph


# configuration
st.set_page_config(
    page_title="Brazilian E-Commerce Dataset Olist Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPH_JSON_PATH = os.path.join(RESULTS_DIR, "graph.json")

NODE_COLORS = {
    "customer": "#4C78A8",
    "order": "#F58518",
    "product": "#54A24B",
    "category": "#E45756",
    "category_original": "#B279A2",
    "seller": "#72B7B2",
    "payment": "#FF9DA6",
    "review": "#9D755D",
    "sentiment": "#EDC948",
    "city": "#59A14F",
    "state": "#76B7B2",
    "geographic_cluster": "#AF7AA1",
}
DEFAULT_NODE_COLOR = "#9E9E9E"

EDGE_COLORS = {
    "PLACED": "#4C78A8",
    "CONTAINS": "#F58518",
    "HAS_PAYMENT": "#E45756",
    "HAS_REVIEW": "#B279A2",
    "HAS_SENTIMENT": "#EDC948",
    "HAS_NLP_SENTIMENT": "#FF9DA6",
    "OFFERED_BY": "#54A24B",
    "BELONGS_TO": "#72B7B2",
    "TRANSLATED_FROM": "#AF7AA1",
    "LOCATED_IN": "#59A14F",
    "BELONGS_TO_CLUSTER": "#AF7AA1",
    "CUSTOMER_INTERACTED_WITH_PRODUCT": "#2F5597",
    "CUSTOMER_ASSOCIATED_WITH_CATEGORY": "#7030A0",
    "CUSTOMER_ASSOCIATED_WITH_SELLER": "#38761D",
    "ORDER_HAS_INFERRED_NLP_SENTIMENT": "#C55A11",
}
DEFAULT_EDGE_COLOR = "#999999"


# graph loading
@st.cache_data
def load_graph():
    if not os.path.exists(GRAPH_JSON_PATH):
        return None

    with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def create_networkx_graph(data):
    G = nx.DiGraph()

    if not data:
        return G

    for node in data.get("nodes", []):
        node_id = node.get("id")

        if node_id is None:
            continue

        attributes = {
            key: value
            for key, value in node.items()
            if key != "id"
        }
        G.add_node(node_id, **attributes)

    for edge in data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")

        if source is None or target is None:
            continue

        attributes = {
            key: value
            for key, value in edge.items()
            if key not in {"source", "target"}
        }
        G.add_edge(source, target, **attributes)

    return G


# search
@st.cache_resource
def create_search_index(_G):
    index = []

    for node in _G.nodes():
        attributes = _G.nodes[node]

        searchable_text = " ".join(
            str(value)
            for value in [
                node,
                attributes.get("display_name", ""),
                attributes.get("type", ""),
                attributes.get("unique_id", ""),
                attributes.get("customer_unique_id", ""),
                attributes.get("city", ""),
                attributes.get("state", ""),
            ]
            if value is not None
        ).lower()

        index.append((str(node), searchable_text))

    return index


def search_nodes(index, query, limit=50):
    query = str(query).lower().strip()

    if not query:
        return []

    results = []

    for node_id, searchable in index:
        if query in searchable:
            results.append(node_id)

            if len(results) >= limit:
                break

    return results


# saved analytics
@st.cache_data
def load_results():
    result_files = [
        "customer_clv.csv",
        "repeat_customers.csv",
        "product_popularity.csv",
        "category_sales.csv",
        "seller_performance_analysis.csv",
        "monthly_revenue.csv",
    ]

    results = {}

    for filename in result_files:
        path = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(path):
            continue

        try:
            results[filename] = pd.read_csv(path)
        except Exception:
            results[filename] = pd.DataFrame()

    return results


def get_result_title(filename):
    titles = {
        "customer_clv.csv": "Historical Customer Spending",
        "repeat_customers.csv": "Repeat Customers",
        "product_popularity.csv": "Product Popularity",
        "category_sales.csv": "Category Sales",
        "seller_performance_analysis.csv": "Seller Performance Analysis",
        "monthly_revenue.csv": "Monthly Payment Value",
    }

    return titles.get(
        filename,
        filename.replace(".csv", "").replace("_", " ").title(),
    )


# graph metrics
@st.cache_data
def calculate_graph_metrics(_G):
    metrics = {
        "nodes": _G.number_of_nodes(),
        "edges": _G.number_of_edges(),
        "density": nx.density(_G),
        "connected_components": (
            nx.number_weakly_connected_components(_G)
            if _G.number_of_nodes()
            else 0
        ),
    }

    degrees = dict(_G.degree())

    if degrees:
        metrics["average_degree"] = (
            sum(degrees.values()) / len(degrees)
        )
        metrics["top_nodes"] = sorted(
            degrees.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    else:
        metrics["average_degree"] = 0
        metrics["top_nodes"] = []

    metrics["inferred_edges"] = sum(
        1
        for _, _, data in _G.edges(data=True)
        if data.get("inferred", False)
    )
    metrics["explicit_edges"] = (
        metrics["edges"] - metrics["inferred_edges"]
    )

    return metrics


# node helpers
def get_node_type(G, node):
    return str(G.nodes[node].get("type", "unknown"))


def get_node_label(G, node):
    data = G.nodes[node]
    node_type = str(data.get("type", "unknown"))

    if node_type == "customer":
        city = str(data.get("city", "")).strip()
        state = str(data.get("state", "")).strip()
        unique_id = str(
            data.get(
                "unique_id",
                data.get("customer_unique_id", ""),
            )
        ).strip()

        if unique_id:
            short_id = unique_id[:8]
            parts = ["Customer"]

            if city:
                parts.append(city)

            if state:
                parts.append(state)

            parts.append(short_id)
            label = " | ".join(parts)
        else:
            label = data.get("display_name", node)
    else:
        label = data.get("display_name", node)

    label = str(label)

    return label if len(label) <= 55 else label[:52] + "..."


def get_node_hover_title(G, node):
    data = G.nodes[node]
    node_type = str(data.get("type", "unknown"))
    label = get_node_label(G, node)

    lines = [
        f"<b>{label}</b>",
        f"Type: {node_type}",
        f"Degree: {G.degree(node)}",
        f"Node ID: {node}",
    ]

    if node_type == "customer":
        unique_id = data.get(
            "unique_id",
            data.get("customer_unique_id"),
        )

        if unique_id:
            lines.append(f"Persistent customer ID: {unique_id}")

        if data.get("city"):
            lines.append(f"City: {data['city']}")

        if data.get("state"):
            lines.append(f"State: {data['state']}")

    useful_attributes = [
        "product_id",
        "seller_id",
        "order_id",
        "category",
        "payment_type",
        "installments",
        "value",
        "score",
        "sentiment",
        "nlp_sentiment",
        "nlp_polarity",
        "source",
    ]

    for attribute in useful_attributes:
        if attribute in data:
            lines.append(f"{attribute}: {data[attribute]}")

    return "<br>".join(lines)


def get_node_color(node_type):
    return NODE_COLORS.get(node_type, DEFAULT_NODE_COLOR)


def get_edge_color(relation):
    return EDGE_COLORS.get(relation, DEFAULT_EDGE_COLOR)


def get_available_entity_types(G):
    types = {
        str(G.nodes[node].get("type"))
        for node in G.nodes()
        if G.nodes[node].get("type")
    }
    return sorted(types)


def get_entities_by_type(G, entity_type):
    entities = [
        node
        for node in G.nodes()
        if str(G.nodes[node].get("type", "")) == entity_type
    ]

    return sorted(
        entities,
        key=lambda node: get_node_label(G, node).lower(),
    )


def create_entity_selection_map(G, nodes):
    selection_map = {}

    for node in nodes:
        label = get_node_label(G, node)
        node_type = get_node_type(G, node)
        base_label = f"{label} [{node_type}]"
        selection_label = base_label
        counter = 2

        while selection_label in selection_map:
            selection_label = f"{base_label} ({counter})"
            counter += 1

        selection_map[selection_label] = node

    return selection_map


# graph subgraphs
def build_representative_subgraph(
    G,
    max_nodes=1000,
    max_depth=10,
    random_seed=None,
):
    if G.number_of_nodes() == 0:
        return nx.DiGraph()

    max_nodes = min(int(max_nodes), G.number_of_nodes())
    max_depth = max(1, int(max_depth))

    preferred_types = {
        "customer",
        "order",
        "product",
        "category",
        "category_original",
        "seller",
        "payment",
        "review",
        "sentiment",
        "city",
        "state",
        "geographic_cluster",
    }

    anchor_types = {
        "order",
        "product",
        "customer",
        "seller",
    }

    rng = random.Random(random_seed)
    anchor_candidates = []

    for node_type in anchor_types:
        candidates = [
            node
            for node in G.nodes()
            if str(G.nodes[node].get("type", "")) == node_type
        ]

        ranked = sorted(
            candidates,
            key=lambda node: G.degree(node),
            reverse=True,
        )

        anchor_candidates.extend(
            ranked[:min(len(ranked), 500)]
        )

    if not anchor_candidates:
        anchor_candidates = list(G.nodes())

    anchor = rng.choice(anchor_candidates)
    selected_nodes = {anchor}
    selected_types = {
        str(G.nodes[anchor].get("type", "unknown"))
    }
    frontier = {anchor}

    for _ in range(max_depth):
        if not frontier or len(selected_nodes) >= max_nodes:
            break

        candidates = set()

        for node in frontier:
            candidates.update(G.successors(node))
            candidates.update(G.predecessors(node))

        candidates.difference_update(selected_nodes)

        if not candidates:
            break

        missing_types = preferred_types - selected_types

        def candidate_score(node):
            node_type = str(
                G.nodes[node].get("type", "unknown")
            )
            degree = G.degree(node)
            diversity_bonus = 10000 if node_type in missing_types else 0
            type_bonus = 100 if node_type in preferred_types else 0
            degree_score = min(degree, 500)
            random_jitter = rng.random() * 250

            return (
                diversity_bonus,
                type_bonus,
                degree_score,
                random_jitter,
                str(node),
            )

        ranked = sorted(
            candidates,
            key=candidate_score,
            reverse=True,
        )

        remaining = max_nodes - len(selected_nodes)
        chosen = []
        chosen_types = set()

        for candidate in ranked:
            node_type = str(
                G.nodes[candidate].get("type", "unknown")
            )

            if (
                node_type not in selected_types
                and node_type not in chosen_types
            ):
                chosen.append(candidate)
                chosen_types.add(node_type)

                if len(chosen) >= remaining:
                    break

        if len(chosen) < remaining:
            for candidate in ranked:
                if candidate not in chosen:
                    chosen.append(candidate)

                if len(chosen) >= remaining:
                    break

        if not chosen:
            break

        selected_nodes.update(chosen)
        selected_types.update(
            str(G.nodes[node].get("type", "unknown"))
            for node in chosen
        )
        frontier = set(chosen)

    return G.subgraph(selected_nodes).copy()


def build_local_subgraph(
    G,
    selected_node,
    depth=1,
    max_nodes=100,
):
    if selected_node not in G:
        return nx.DiGraph()

    selected_nodes = {selected_node}
    frontier = {selected_node}

    for _ in range(depth):
        next_frontier = set()

        for node in frontier:
            next_frontier.update(G.successors(node))
            next_frontier.update(G.predecessors(node))

        next_frontier.difference_update(selected_nodes)

        if not next_frontier:
            break

        remaining = max_nodes - len(selected_nodes)

        if remaining <= 0:
            break

        ranked = sorted(
            next_frontier,
            key=lambda node: G.degree(node),
            reverse=True,
        )

        selected_frontier = set(ranked[:remaining])
        selected_nodes.update(selected_frontier)
        frontier = selected_frontier

    return G.subgraph(selected_nodes).copy()


def build_entity_type_overview_subgraph(
    G,
    entity_type,
    max_nodes=500,
):
    if G.number_of_nodes() == 0:
        return nx.DiGraph()

    candidates = get_entities_by_type(G, entity_type)

    if not candidates:
        return nx.DiGraph()

    ranked = sorted(
        candidates,
        key=lambda node: G.degree(node),
        reverse=True,
    )

    pool = ranked[:min(len(ranked), 200)]
    seed_count = min(20, len(pool), max_nodes)
    rng = random.Random(42)
    seed_nodes = rng.sample(pool, seed_count)

    selected_nodes = set(seed_nodes)
    frontier = set(seed_nodes)

    while frontier and len(selected_nodes) < max_nodes:
        candidates_next = set()

        for node in frontier:
            candidates_next.update(G.successors(node))
            candidates_next.update(G.predecessors(node))

        candidates_next.difference_update(selected_nodes)

        if not candidates_next:
            break

        ranked_candidates = sorted(
            candidates_next,
            key=lambda node: G.degree(node),
            reverse=True,
        )

        remaining = max_nodes - len(selected_nodes)
        next_frontier = set(
            ranked_candidates[:remaining]
        )

        selected_nodes.update(next_frontier)
        frontier = next_frontier

    return G.subgraph(selected_nodes).copy()


# pyvis
def apply_pyvis_options(net, max_nodes):
    if max_nodes >= 1000:
        stabilization_iterations = 150
        gravitational_constant = -4000
        spring_length = 120
    else:
        stabilization_iterations = 250
        gravitational_constant = -5000
        spring_length = 150

    net.set_options(
        f"""
        {{
          "nodes": {{
            "font": {{"size": 13, "face": "Arial"}},
            "borderWidth": 2
          }},
          "edges": {{
            "font": {{
              "size": 10,
              "align": "middle",
              "background": "white"
            }},
            "smooth": {{"enabled": true, "type": "dynamic"}},
            "arrows": {{
              "to": {{"enabled": true, "scaleFactor": 0.7}}
            }}
          }},
          "physics": {{
            "enabled": true,
            "barnesHut": {{
              "gravitationalConstant": {gravitational_constant},
              "centralGravity": 0.2,
              "springLength": {spring_length},
              "springConstant": 0.04,
              "damping": 0.25
            }},
            "stabilization": {{
              "enabled": true,
              "iterations": {stabilization_iterations}
            }}
          }},
          "interaction": {{
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
          }}
        }}
        """
    )


def create_pyvis_graph_from_subgraph(
    G,
    local_graph,
    selected_node=None,
    max_nodes=500,
):
    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#222222",
        notebook=False,
    )

    for node in local_graph.nodes():
        node_type = get_node_type(G, node)
        degree = G.degree(node)

        net.add_node(
            str(node),
            label=get_node_label(G, node),
            title=get_node_hover_title(G, node),
            color=get_node_color(node_type),
            size=35 if node == selected_node else min(
                15 + degree * 1.2,
                30,
            ),
            borderWidth=2,
            shape="dot",
        )

    for source, target, edge_data in local_graph.edges(data=True):
        relation = str(edge_data.get("relation", ""))
        inferred = bool(edge_data.get("inferred", False))

        title = relation

        if inferred:
            rule = edge_data.get("inference_rule", "")
            title = "Inferred relationship"

            if rule:
                title += f"<br>Rule: {rule}"

        net.add_edge(
            str(source),
            str(target),
            label=relation,
            title=title,
            color=get_edge_color(relation),
            arrows="to",
            width=2,
            dashes=inferred,
        )

    apply_pyvis_options(net, max_nodes)
    return net


def create_pyvis_graph(
    G,
    selected_node=None,
    depth=1,
    max_nodes=100,
    overview=False,
    random_seed=None,
):
    if G.number_of_nodes() == 0:
        return Network(
            height="700px",
            width="100%",
            directed=True,
        )

    if overview:
        local_graph = build_representative_subgraph(
            G,
            max_nodes=max_nodes,
            max_depth=depth,
            random_seed=random_seed,
        )
    elif selected_node is not None:
        local_graph = build_local_subgraph(
            G,
            selected_node,
            depth=depth,
            max_nodes=max_nodes,
        )
    else:
        local_graph = nx.DiGraph()

    return create_pyvis_graph_from_subgraph(
        G,
        local_graph,
        selected_node=selected_node,
        max_nodes=max_nodes,
    )


# relationship and entity information
def get_relationship_table(G, selected_node):
    if selected_node not in G:
        return pd.DataFrame()

    rows = []

    for target in G.successors(selected_node):
        edge_data = G.get_edge_data(
            selected_node,
            target,
            default={},
        )
        target_data = G.nodes[target]

        rows.append({
            "direction": "OUTGOING",
            "relation": edge_data.get("relation", ""),
            "inferred": edge_data.get("inferred", False),
            "connected_node": target,
            "connected_type": target_data.get("type", ""),
            "connected_name": get_node_label(G, target),
        })

    for source in G.predecessors(selected_node):
        edge_data = G.get_edge_data(
            source,
            selected_node,
            default={},
        )
        source_data = G.nodes[source]

        rows.append({
            "direction": "INCOMING",
            "relation": edge_data.get("relation", ""),
            "inferred": edge_data.get("inferred", False),
            "connected_node": source,
            "connected_type": source_data.get("type", ""),
            "connected_name": get_node_label(G, source),
        })

    return pd.DataFrame(rows)


def get_entity_information(G, selected_node):
    if selected_node not in G:
        return pd.DataFrame()

    return pd.DataFrame([
        {"attribute": key, "value": value}
        for key, value in G.nodes[selected_node].items()
    ])


def display_dataframe(dataframe):
    if dataframe is None:
        return

    if dataframe.empty:
        st.info("No matching records were found.")
        return

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def display_graph_legend():
    st.markdown("### Node types")

    columns = st.columns(4)

    for index, (node_type, color) in enumerate(
        NODE_COLORS.items()
    ):
        with columns[index % 4]:
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:8px;
                    margin-bottom:6px;
                ">
                    <div style="
                        width:14px;
                        height:14px;
                        border-radius:50%;
                        background:{color};
                        border:1px solid #555;
                    "></div>
                    <span>{node_type}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# application
def main():
    if "kg_overview_seed" not in st.session_state:
        st.session_state["kg_overview_seed"] = (
            random.SystemRandom().randrange(
                1,
                2**31 - 1,
            )
        )

    overview_seed = st.session_state["kg_overview_seed"]

    st.title("Olist Knowledge Graph Explorer")
    st.caption(
        "Knowledge Graph for E-commerce Customer "
        "Behavior and Financial Analysis"
    )

    data = load_graph()

    if data is None:
        st.error(
            "results/graph.json was not found. "
            "Run main.py first to construct and export "
            "the Knowledge Graph."
        )
        st.stop()

    G = create_networkx_graph(data)
    search_index = create_search_index(G)
    results = load_results()
    metrics = calculate_graph_metrics(G)

    # sidebar
    st.sidebar.title("Knowledge Graph Service")

    page = st.sidebar.radio(
        "Select a service",
        [
            "KG Service Overview",
            "Ask the Knowledge Graph",
            "Graph Explorer",
            "Analytics",
            "Search Entity",
            "Graph Metrics",
        ],
    )

    # service overview
    if page == "KG Service Overview":
        st.header("Knowledge Graph Service")

        st.write(
            "This application exposes the Olist e-commerce "
            "Knowledge Graph through deterministic analytical "
            "queries, entity search, graph exploration and "
            "business analytics."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("KG Nodes", f"{G.number_of_nodes():,}")

        with col2:
            st.metric("KG Relationships (Edges)", f"{G.number_of_edges():,}")

        with col3:
            st.metric(
                "Average Degree",
                f"{metrics['average_degree']:.2f}",
            )

        with col4:
            st.metric(
                "Inferred Relationships",
                f"{metrics['inferred_edges']:,}",
            )

        st.subheader("Available KG services")

        st.markdown(
            """
            **Ask the Knowledge Graph**

            Enter analytical questions about customers,
            spending, orders, products, categories, sellers,
            payments and reviews.

            **Graph Explorer**

            Explore representative connected subgraphs or
            inspect a selected entity and its multi-hop
            neighbourhood.

            **Analytics**

            View predefined business analytics generated
            from the project datasets.

            **Search Entity**

            Search the KG and inspect entity attributes and
            incoming/outgoing relationships.

            **Graph Metrics**

            Inspect structural properties and relationship
            distributions of the exported KG.
            """
        )

        st.subheader("Graph visualization")
        display_graph_legend()

    # analytical query service
    elif page == "Ask the Knowledge Graph":
        st.header(
            ":material/search: Ask the Knowledge Graph"
        )

        st.write(
            "Enter an analytical question in natural language. "
            "The deterministic query service identifies a supported "
            "intent and answers it using KG traversal and project "
            "analytics. This is not an LLM-based interface."
        )

        with st.form("kg_query_form"):
            question = st.text_input(
                "Your question",
                placeholder="Which customers spent the most?",
            )

            st.caption(
                "Examples: Which products are most popular? · "
                "What is the average order value? · "
                "Show repeat customers · "
                "Show top 25 repeat customers · "
                "Which categories have the most orders?"
            )

            submitted = st.form_submit_button(
                "Ask",
                type="primary",
            )

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner(
                    "Querying the Knowledge Graph..."
                ):
                    result = ask_knowledge_graph(
                        G,
                        question,
                    )

                if result.get("supported", False):
                    st.success(
                        "Question answered from the "
                        "Knowledge Graph."
                    )

                    st.subheader("Answer")
                    st.write(result.get("answer", ""))

                    result_data = result.get("data")

                    if result_data is not None:
                        st.subheader("Results")
                        display_dataframe(result_data)

                    paths = result.get("paths", [])

                    if paths:
                        st.subheader(
                            "How the Knowledge Graph was used"
                        )

                        for path in paths:
                            st.code(path, language="text")

                    intent = result.get("intent")

                    if intent:
                        st.caption(
                            f"Detected analytical intent: {intent}"
                        )
                else:
                    st.warning(
                        result.get(
                            "answer",
                            "The question could not be answered.",
                        )
                    )

    # graph explorer
    elif page == "Graph Explorer":
        st.header(":material/hub: Graph Explorer")

        st.write(
            "Explore a representative connected KG overview, "
            "a type-focused graph, or a specific entity and "
            "its bounded multi-hop neighbourhood."
        )

        st.subheader("Explore an entity")

        available_entity_types = get_available_entity_types(G)

        preferred_order = [
            "customer",
            "order",
            "product",
            "seller",
            "category",
            "category_original",
            "payment",
            "review",
            "sentiment",
            "city",
            "state",
            "geographic_cluster",
        ]

        ordered_entity_types = [
            entity_type
            for entity_type in preferred_order
            if entity_type in available_entity_types
        ]

        ordered_entity_types.extend(
            entity_type
            for entity_type in available_entity_types
            if entity_type not in ordered_entity_types
        )

        general_option = "Default — General KG Overview"
        entity_type_options = [
            general_option,
            *ordered_entity_types,
        ]

        selected_entity_type = st.selectbox(
            "Entity type",
            entity_type_options,
            index=0,
            key="graph_explorer_entity_type",
        )

        general_entity_view = (
            selected_entity_type == general_option
        )

        selected_node = None

        if general_entity_view:
            st.info(
                "General KG overview selected. The graph will "
                "show connected examples from multiple entity types."
            )
        else:
            entity_filter = st.text_input(
                "Filter entities",
                placeholder=(
                    "e.g. franca, SP, customer ID, product ID..."
                ),
                key="graph_explorer_entity_filter",
            )

            entities = get_entities_by_type(
                G,
                selected_entity_type,
            )

            if entity_filter.strip():
                filter_text = entity_filter.lower().strip()
                filtered_entities = []

                for node in entities:
                    node_data = G.nodes[node]

                    searchable = " ".join(
                        str(value)
                        for value in [
                            node,
                            get_node_label(G, node),
                            node_data.get("display_name", ""),
                            node_data.get("unique_id", ""),
                            node_data.get("customer_unique_id", ""),
                            node_data.get("city", ""),
                            node_data.get("state", ""),
                            node_data.get("product_id", ""),
                            node_data.get("seller_id", ""),
                            node_data.get("order_id", ""),
                        ]
                        if value is not None
                    ).lower()

                    if filter_text in searchable:
                        filtered_entities.append(node)

                entities = filtered_entities

            displayed_entities = entities[:500]

            if len(entities) > 500:
                st.info(
                    f"{len(entities):,} entities match the filter. "
                    "Showing the first 500."
                )

            selection_map = create_entity_selection_map(
                G,
                displayed_entities,
            )

            specific_option = (
                "Default — General "
                + selected_entity_type.replace("_", " ").title()
                + " View"
            )

            selected_label = st.selectbox(
                "Select entity",
                [specific_option, *selection_map.keys()],
                index=0,
                key="graph_explorer_entity_selection",
            )

            if selected_label != specific_option:
                selected_node = selection_map[selected_label]
            else:
                st.info(
                    f"General {selected_entity_type.replace('_', ' ')} "
                    "view selected."
                )

        st.divider()
        st.subheader("Visualization settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            depth = st.slider(
                "Neighborhood depth",
                1,
                20,
                3,
            )

        with col2:
            max_nodes = st.slider(
                "Maximum graph nodes",
                50,
                10000,
                500,
                step=50,
            )

        with col3:
            show_legend = st.checkbox(
                "Show entity legend",
                value=True,
                key="entity_legend",
            )

        if max_nodes > 1000:
            st.warning(
                "Large visualization selected. Rendering may "
                "take longer and appear visually dense."
            )

        if general_entity_view:
            st.subheader(
                "Representative Knowledge Graph overview"
            )

            local_overview = build_representative_subgraph(
                G,
                max_nodes=max_nodes,
                max_depth=depth,
                random_seed=overview_seed,
            )

            st.caption(
                f"Showing {local_overview.number_of_nodes():,} nodes "
                f"and {local_overview.number_of_edges():,} relationships (edges)."
            )

            net = create_pyvis_graph(
                G,
                depth=depth,
                max_nodes=max_nodes,
                overview=True,
                random_seed=overview_seed,
            )

            try:
                st.components.v1.html(
                    net.generate_html(),
                    height=750,
                    scrolling=True,
                )
            except Exception as error:
                st.error(f"Visualization failed: {error}")

            if show_legend:
                display_graph_legend()

            overview_type_counts = (
                pd.Series(
                    [
                        G.nodes[node].get("type", "unknown")
                        for node in local_overview.nodes()
                    ]
                )
                .value_counts()
                .rename_axis("type")
                .reset_index(name="nodes")
            )

            st.subheader("Overview entity types")
            display_dataframe(overview_type_counts)

        elif selected_node is None:
            st.subheader(
                f"General "
                f"{selected_entity_type.replace('_', ' ').title()} "
                "view"
            )

            type_overview = build_entity_type_overview_subgraph(
                G,
                selected_entity_type,
                max_nodes=max_nodes,
            )

            st.caption(
                f"Showing {type_overview.number_of_nodes():,} nodes "
                f"and {type_overview.number_of_edges():,} relationships (edges)."
            )

            net = create_pyvis_graph_from_subgraph(
                G,
                type_overview,
                max_nodes=max_nodes,
            )

            try:
                st.components.v1.html(
                    net.generate_html(),
                    height=750,
                    scrolling=True,
                )
            except Exception as error:
                st.error(f"Visualization failed: {error}")

            if show_legend:
                display_graph_legend()

        else:
            node_data = G.nodes[selected_node]
            node_type = node_data.get("type", "unknown")

            st.subheader("Selected entity")

            st.markdown(
                f"""
                **{get_node_label(G, selected_node)}**

                Type: `{node_type}` ·
                Degree: `{G.degree(selected_node)}` ·
                ID: `{selected_node}`
                """
            )

            local_graph = build_local_subgraph(
                G,
                selected_node,
                depth=depth,
                max_nodes=max_nodes,
            )

            st.caption(
                f"Showing {local_graph.number_of_nodes():,} nodes "
                f"and {local_graph.number_of_edges():,} relationships (edges)."
            )

            net = create_pyvis_graph(
                G,
                selected_node=selected_node,
                depth=depth,
                max_nodes=max_nodes,
            )

            try:
                st.components.v1.html(
                    net.generate_html(),
                    height=750,
                    scrolling=True,
                )
            except Exception as error:
                st.error(f"Visualization failed: {error}")

            if show_legend:
                display_graph_legend()

            st.subheader("Relationships (edges)")

            relationship_df = get_relationship_table(
                G,
                selected_node,
            )

            display_dataframe(relationship_df)

    # analytics
    elif page == "Analytics":
        st.header(":material/analytics: Business Analytics")

        if not results:
            st.info("No saved analytics were found.")
        else:
            selected = st.selectbox(
                "Select analysis",
                list(results.keys()),
                format_func=get_result_title,
            )

            st.subheader(get_result_title(selected))
            display_dataframe(results[selected])

    # entity search
    elif page == "Search Entity":
        st.header(":material/search: Search Knowledge Graph Entity")

        st.write(
            "Search for a customer, product, order, seller, "
            "category, city or other KG entity."
        )

        query = st.text_input(
            "Search",
            placeholder="Enter an entity name, ID, city, category...",
        )

        if query:
            matches = search_nodes(
                search_index,
                query,
                limit=100,
            )

            if not matches:
                st.info("No matching entities found.")
            else:
                selection_map = create_entity_selection_map(
                    G,
                    matches,
                )

                selected_label = st.selectbox(
                    "Select entity",
                    list(selection_map.keys()),
                    key="search_entity_selection",
                )

                selected_node = selection_map[selected_label]
                data = G.nodes[selected_node]

                st.subheader("Entity information")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Type",
                        str(data.get("type", "unknown")),
                    )

                with col2:
                    st.metric(
                        "Degree",
                        G.degree(selected_node),
                    )

                with col3:
                    st.metric(
                        "Relationships",
                        G.in_degree(selected_node)
                        + G.out_degree(selected_node),
                    )

                display_dataframe(
                    get_entity_information(
                        G,
                        selected_node,
                    )
                )

                st.subheader("Incoming and outgoing relationships")

                display_dataframe(
                    get_relationship_table(
                        G,
                        selected_node,
                    )
                )

                st.subheader("Entity neighborhood")

                col1, col2 = st.columns(2)

                with col1:
                    search_depth = st.slider(
                        "Depth",
                        1,
                        2,
                        1,
                        key="search_graph_depth",
                    )

                with col2:
                    search_max_nodes = st.slider(
                        "Maximum nodes",
                        20,
                        150,
                        60,
                        step=10,
                        key="search_graph_nodes",
                    )

                net = create_pyvis_graph(
                    G,
                    selected_node=selected_node,
                    depth=search_depth,
                    max_nodes=search_max_nodes,
                )

                try:
                    st.components.v1.html(
                        net.generate_html(),
                        height=650,
                        scrolling=True,
                    )
                except Exception as error:
                    st.error(f"Visualization failed: {error}")

                display_graph_legend()

    # metrics
    elif page == "Graph Metrics":
        st.header(":material/monitoring: Knowledge Graph Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nodes", f"{metrics['nodes']:,}")

        with col2:
            st.metric("Relationships", f"{metrics['edges']:,}")

        with col3:
            st.metric(
                "Average degree",
                f"{metrics['average_degree']:.2f}",
            )

        with col4:
            st.metric(
                "Inferred relationships",
                f"{metrics['inferred_edges']:,}",
            )

        st.metric(
            "Weakly connected components",
            metrics["connected_components"],
        )

        st.subheader("Node type distribution")

        node_type_counts = (
            pd.Series(
                [
                    G.nodes[node].get("type", "unknown")
                    for node in G.nodes()
                ]
            )
            .value_counts()
            .rename_axis("type")
            .reset_index(name="count")
        )

        display_dataframe(node_type_counts)

        st.subheader("Relationship type distribution")

        relationship_counts = (
            pd.Series(
                [
                    data.get("relation", "unknown")
                    for _, _, data in G.edges(data=True)
                ]
            )
            .value_counts()
            .rename_axis("relation")
            .reset_index(name="count")
        )

        display_dataframe(relationship_counts)

        st.subheader("Explicit vs inferred relationships")

        edge_origin = pd.DataFrame([
            {
                "edge_type": "Explicit",
                "count": metrics["explicit_edges"],
            },
            {
                "edge_type": "Inferred",
                "count": metrics["inferred_edges"],
            },
        ])

        display_dataframe(edge_origin)

        st.subheader("Most connected entities")

        rows = []

        for node, degree in metrics["top_nodes"]:
            rows.append({
                "node": node,
                "display_name": get_node_label(G, node),
                "degree": degree,
                "type": G.nodes[node].get("type", ""),
            })

        display_dataframe(pd.DataFrame(rows))

    # footer
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Olist Knowledge Graph · NetworkX · Streamlit"
    )


if __name__ == "__main__":
    main()