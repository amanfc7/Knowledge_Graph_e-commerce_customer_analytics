import networkx as nx
import pandas as pd

def build_graph(customers, orders, order_items, products, payments, category):

    G = nx.DiGraph()

    # -----------------------------
    # SCHEMA LAYER (IMPORTANT FIX)
    # -----------------------------
    SCHEMA = {
        "customer": "entity",
        "order": "event",
        "product": "entity",
        "payment": "financial_event",
        "category": "concept"
    }

    # -----------------------------
    # CUSTOMERS (ENTITY NODES)
    # -----------------------------
    for _, row in customers.iterrows():
        G.add_node(
            row["customer_id"],
            type="customer",
            schema_type=SCHEMA["customer"]
        )

    # -----------------------------
    # ORDERS (EVENT NODES)
    # -----------------------------
    # -----------------------
# Orders → Products + Sellers (IMPORTANT KG RELATIONS)
# -----------------------
    for _, row in order_items.iterrows():
        G.add_node(row["product_id"], type="product")
        G.add_node(row["seller_id"], type="seller")  # NEW NODE TYPE

    # Order → Product
        G.add_edge(row["order_id"], row["product_id"], relation="CONTAINS")

    # Order → Seller  ⭐ IMPORTANT
        G.add_edge(row["order_id"], row["seller_id"], relation="SOLD_BY")

    # Product → Seller (optional but VERY useful)
        G.add_edge(row["product_id"], row["seller_id"], relation="OFFERED_BY")

    # -----------------------------
    # PRODUCTS (ENTITY NODES)
    # -----------------------------
    for _, row in products.iterrows():
        if pd.notna(row["product_id"]):
            G.add_node(
                row["product_id"],
                type="product",
                schema_type=SCHEMA["product"]
            )


    # -----------------------------
    # PAYMENTS (REIFICATION MODEL)
    # -----------------------------
    for _, row in payments.iterrows():
        payment_node = f"payment_{row['order_id']}_{row.name}"

        G.add_node(
            payment_node,
            type="payment",
            schema_type=SCHEMA["payment"],
            value=row["payment_value"]
        )

        G.add_edge(row["order_id"], payment_node, relation="HAS_PAYMENT")

    # -----------------------------
    # CATEGORY MAPPING (NORMALIZED)
    # -----------------------------
    products = products.merge(category, on="product_category_name", how="left")

    for _, row in products.iterrows():
        if pd.notna(row["product_category_name_english"]):

            G.add_node(
                row["product_category_name_english"],
                type="category",
                schema_type=SCHEMA["category"]
            )

            G.add_edge(
                row["product_id"],
                row["product_category_name_english"],
                relation="BELONGS_TO"
            )

    return G