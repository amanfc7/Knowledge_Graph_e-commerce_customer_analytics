import networkx as nx
import pandas as pd


def build_graph(customers, orders, order_items, products, payments, sellers, category):

    G = nx.DiGraph()

    # ==========================================================
    # KNOWLEDGE GRAPH SCHEMA
    # ==========================================================

    SCHEMA = {
        "customer": "entity",
        "order": "event",
        "product": "entity",
        "seller": "entity",
        "payment": "financial_event",
        "category": "concept",
        "state": "location"
    }

    # ==========================================================
    # CUSTOMER NODES
    # ==========================================================

    for _, row in customers.iterrows():

        G.add_node(
            row["customer_id"],
            type="customer",
            schema_type=SCHEMA["customer"],
            city=row["customer_city"],
            state=row["customer_state"]
        )

        state_node = f"STATE_{row['customer_state']}"

        G.add_node(
            state_node,
            type="state",
            schema_type=SCHEMA["state"]
        )

        G.add_edge(
            row["customer_id"],
            state_node,
            relation="LOCATED_IN"
        )

    # ==========================================================
    # ORDER NODES
    # ==========================================================

    for _, row in orders.iterrows():

        G.add_node(
            row["order_id"],
            type="order",
            schema_type=SCHEMA["order"],
            status=row["order_status"]
        )

        G.add_edge(
            row["customer_id"],
            row["order_id"],
            relation="PLACED"
        )

    # ==========================================================
    # SELLER NODES
    # ==========================================================

    for _, row in sellers.iterrows():

        G.add_node(
            row["seller_id"],
            type="seller",
            schema_type=SCHEMA["seller"],
            city=row["seller_city"],
            state=row["seller_state"]
        )

        state_node = f"STATE_{row['seller_state']}"

        G.add_node(
            state_node,
            type="state",
            schema_type=SCHEMA["state"]
        )

        G.add_edge(
            row["seller_id"],
            state_node,
            relation="LOCATED_IN"
        )

    # ==========================================================
    # PRODUCTS
    # ==========================================================


    for _, row in products.iterrows():

        G.add_node(
            row["product_id"],
            type="product",
            schema_type=SCHEMA["product"],
            weight=row["product_weight_g"],
            length=row["product_length_cm"],
            height=row["product_height_cm"],
            width=row["product_width_cm"]
        )

        if pd.notna(row["product_category_name_english"]):

            category_name = row["product_category_name_english"]

            G.add_node(
                category_name,
                type="category",
                schema_type=SCHEMA["category"]
            )

            G.add_edge(
                row["product_id"],
                category_name,
                relation="BELONGS_TO"
            )

    # ==========================================================
    # ORDER ITEMS
    # ==========================================================

    for _, row in order_items.iterrows():

        G.add_edge(
            row["order_id"],
            row["product_id"],
            relation="CONTAINS"
        )

        G.add_edge(
            row["order_id"],
            row["seller_id"],
            relation="SOLD_BY"
        )

        G.add_edge(
            row["product_id"],
            row["seller_id"],
            relation="OFFERED_BY"
        )

    # ==========================================================
    # PAYMENTS
    # ==========================================================

    for idx, row in payments.iterrows():

        payment_node = f"PAYMENT_{idx}"

        G.add_node(
            payment_node,
            type="payment",
            schema_type=SCHEMA["payment"],
            payment_type=row["payment_type"],
            installments=row["payment_installments"],
            value=row["payment_value"]
        )

        G.add_edge(
            row["order_id"],
            payment_node,
            relation="HAS_PAYMENT"
        )

    # ==========================================================
    # GRAPH SUMMARY
    # ==========================================================

    print("\n--- KNOWLEDGE GRAPH CREATED ---")
    print("Nodes :", G.number_of_nodes())
    print("Edges :", G.number_of_edges())

    print("\nNode Types")

    node_types = {}

    for _, data in G.nodes(data=True):

        t = data["type"]
        node_types[t] = node_types.get(t, 0) + 1

    for t, c in sorted(node_types.items()):
        print(f"{t:<12}: {c}")

    return G