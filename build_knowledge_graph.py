import networkx as nx

def build_graph(customers, orders, order_items, products, payments, category):

    G = nx.DiGraph()

    # Customers → Orders
    for _, row in customers.iterrows():
        G.add_node(row["customer_id"], type="customer")

    for _, row in orders.iterrows():
        G.add_node(row["order_id"], type="order")
        G.add_edge(row["customer_id"], row["order_id"], relation="PLACED")

    # Orders → Products
    for _, row in order_items.iterrows():
        G.add_node(row["product_id"], type="product")
        G.add_edge(row["order_id"], row["product_id"], relation="CONTAINS")

    # Orders → Payments (financial info)
    for _, row in payments.iterrows():
        G.add_edge(row["order_id"], "payment", relation="HAS_PAYMENT")

    # Products → Category
    products = products.merge(category, on="product_category_name", how="left")

    for _, row in products.iterrows():
        G.add_node(row["product_id"], type="product")
        if pd.notna(row["product_category_name_english"]):
            G.add_node(row["product_category_name_english"], type="category")
            G.add_edge(row["product_id"], row["product_category_name_english"], relation="BELONGS_TO")

    return G