import networkx as nx
import pandas as pd
import os


# =====================================================
# SAVE QUERY RESULTS
# =====================================================

def save_results(name, df):

    os.makedirs(
        "results/queries",
        exist_ok=True
    )

    df.to_csv(
        f"results/queries/{name}.csv",
        index=False
    )


# =====================================================
# HIGH VALUE CUSTOMERS
# =====================================================

def query_high_value_customers(G, limit=10):

    results = []

    for node, attr in G.nodes(data=True):

        if attr.get("type") == "customer":

            orders = list(G.successors(node))

            results.append(

                (
                    node,
                    len(orders),
                    attr.get("city"),
                    attr.get("state")
                )

            )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return pd.DataFrame(

        results[:limit],

        columns=[
            "customer_id",
            "orders",
            "city",
            "state"
        ]

    )


# =====================================================
# POPULAR PRODUCTS
# =====================================================

def query_popular_products(G, limit=10):

    products = []

    for node, attr in G.nodes(data=True):

        if attr.get("type") == "product":

            incoming = list(G.predecessors(node))

            products.append(

                (
                    node,
                    len(incoming)
                )

            )

    products.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return pd.DataFrame(

        products[:limit],

        columns=[
            "product_id",
            "connections"
        ]

    )


# =====================================================
# CENTRAL SELLERS
# =====================================================

def query_central_sellers(G, limit=10):

    sellers = []

    for node, attr in G.nodes(data=True):

        if attr.get("type") == "seller":

            sellers.append(

                (
                    node,
                    G.degree(node),
                    attr.get("city"),
                    attr.get("state")
                )

            )

    sellers.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return pd.DataFrame(

        sellers[:limit],

        columns=[
            "seller_id",
            "degree",
            "city",
            "state"
        ]

    )


# =====================================================
# TOP CATEGORIES
# =====================================================

def query_categories(G):

    categories = []

    for node, attr in G.nodes(data=True):

        if attr.get("type") == "category":

            count = len(list(G.predecessors(node)))

            categories.append(

                (
                    node,
                    count
                )

            )

    df = pd.DataFrame(

        categories,

        columns=[
            "category",
            "products"
        ]

    )

    return df.sort_values(
        "products",
        ascending=False
    )


# =====================================================
# CUSTOMER STATES
# =====================================================

def query_customer_states(G):

    states = {}

    for _, attr in G.nodes(data=True):

        if attr.get("type") == "customer":

            state = attr.get("state")

            states[state] = states.get(state, 0) + 1

    df = pd.DataFrame(

        states.items(),

        columns=[
            "state",
            "customers"
        ]

    )

    return df.sort_values(
        "customers",
        ascending=False
    )


# =====================================================
# PAYMENT TYPES
# =====================================================

def query_payment_methods(G):

    payments = {}

    for _, attr in G.nodes(data=True):

        if attr.get("type") == "payment":

            payment = attr.get("payment_type")

            payments[payment] = payments.get(payment, 0) + 1

    df = pd.DataFrame(

        payments.items(),

        columns=[
            "payment_type",
            "count"
        ]

    )

    return df.sort_values(
        "count",
        ascending=False
    )


# =====================================================
# SELLER STATES
# =====================================================

def query_seller_states(G):

    sellers = {}

    for _, attr in G.nodes(data=True):

        if attr.get("type") == "seller":

            state = attr.get("state")

            sellers[state] = sellers.get(state, 0) + 1

    df = pd.DataFrame(

        sellers.items(),

        columns=[
            "state",
            "seller_count"
        ]

    )

    return df.sort_values(
        "seller_count",
        ascending=False
    )


# =====================================================
# MAIN QUERY ENGINE
# =====================================================

def run_graph_queries(G):

    print("\n==============================")
    print(" KNOWLEDGE GRAPH QUERY ENGINE")
    print("==============================")

    customers = query_high_value_customers(G)

    print("\nTop Customers")
    print(customers)

    save_results(
        "top_customers",
        customers
    )

    products = query_popular_products(G)

    print("\nPopular Products")
    print(products)

    save_results(
        "popular_products",
        products
    )

    sellers = query_central_sellers(G)

    print("\nCentral Sellers")
    print(sellers)

    save_results(
        "central_sellers",
        sellers
    )

    categories = query_categories(G)

    print("\nTop Categories")
    print(categories.head(10))

    save_results(
        "top_categories",
        categories
    )

    customer_states = query_customer_states(G)

    print("\nCustomer Distribution")
    print(customer_states)

    save_results(
        "customer_states",
        customer_states
    )

    payment_methods = query_payment_methods(G)

    print("\nPayment Methods")
    print(payment_methods)

    save_results(
        "payment_methods",
        payment_methods
    )

    seller_states = query_seller_states(G)

    print("\nSeller Distribution")
    print(seller_states)

    save_results(
        "seller_states",
        seller_states
    )

    print("\nAll query results saved to results/queries/")

    return {
        "customers": customers,
        "products": products,
        "sellers": sellers,
        "categories": categories,
        "customer_states": customer_states,
        "payment_methods": payment_methods,
        "seller_states": seller_states
    }