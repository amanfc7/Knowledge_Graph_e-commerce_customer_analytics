import networkx as nx


# =====================================================
# KG QUERY ENGINE
# Datalog-style graph queries
# =====================================================


def query_high_value_customers(G, limit=10):

    results = []

    for node, attr in G.nodes(data=True):

        if attr.get("type") == "customer":

            orders = list(G.successors(node))

            if len(orders) > 0:

                results.append(
                    (
                        node,
                        len(orders)
                    )
                )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results[:limit]



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


    return products[:limit]



def query_central_sellers(G, limit=10):

    sellers = []

    for node, attr in G.nodes(data=True):

        if attr.get("type") == "seller":

            degree = G.degree(node)

            sellers.append(
                (
                    node,
                    degree
                )
            )


    sellers.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return sellers[:limit]



# =====================================================
# MAIN QUERY FUNCTION
# =====================================================

def run_graph_queries(G):

    print("\n--- KG QUERY ENGINE ---")


    print("\nHigh Value Customers")
    print("--------------------")

    for customer, score in query_high_value_customers(G):

        print(
            customer,
            "orders:",
            score
        )



    print("\nPopular Products")
    print("----------------")

    for product, score in query_popular_products(G):

        print(
            product,
            "connections:",
            score
        )



    print("\nCentral Sellers")
    print("---------------")

    for seller, score in query_central_sellers(G):

        print(
            seller,
            "degree:",
            score
        )