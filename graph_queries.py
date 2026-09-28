"""
Deterministic natural-language query service for the e-commerce Knowledge Graph.

KG direction:
    Customer -> PLACED -> Order
    Order -> HAS_PAYMENT -> Payment
    Order -> CONTAINS -> Product
    Product -> OFFERED_BY -> Seller
    Product -> BELONGS_TO -> Category
    Category -> TRANSLATED_FROM -> Original Category
    Order -> HAS_REVIEW -> Review
    Review -> HAS_SENTIMENT -> Sentiment
    Review -> HAS_NLP_SENTIMENT -> Sentiment
    Customer -> LOCATED_IN -> City
    City -> LOCATED_IN -> State
    Seller -> LOCATED_IN -> City

The service uses deterministic intent detection and graph traversal.
"""

import re
import unicodedata
from collections import defaultdict

import pandas as pd


# relationship names
PLACED = "PLACED"
HAS_PAYMENT = "HAS_PAYMENT"
CONTAINS = "CONTAINS"
OFFERED_BY = "OFFERED_BY"
BELONGS_TO = "BELONGS_TO"
HAS_REVIEW = "HAS_REVIEW"
HAS_SENTIMENT = "HAS_SENTIMENT"
HAS_NLP_SENTIMENT = "HAS_NLP_SENTIMENT"
LOCATED_IN = "LOCATED_IN"


# text utilities
def normalize(text):
    """Lowercase, remove accents, and normalize whitespace."""
    text = "" if text is None else str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_any(text, phrases):
    """Check whether any phrase occurs in normalized text."""
    text = normalize(text)
    return any(
        normalize(phrase) in text
        for phrase in phrases
    )


# graph utilities
def nodes_of_type(G, kind):
    return [
        n
        for n, attrs in G.nodes(data=True)
        if attrs.get("type") == kind
    ]


def node_type(G, node):
    return G.nodes[node].get("type")


def display_name(G, node):
    attrs = G.nodes[node]
    return attrs.get(
        "display_name",
        attrs.get("name", str(node)),
    )


def relation_exists(G, source, target, relation):
    """Check whether a directed edge has the requested relation."""
    data = G.get_edge_data(source, target)

    if data is None:
        return False

    if isinstance(data, dict) and "relation" in data:
        return data.get("relation") == relation

    return any(
        isinstance(attrs, dict)
        and attrs.get("relation") == relation
        for attrs in data.values()
    )


def outgoing(G, node, relation=None, kind=None):
    """Return outgoing neighbors filtered by relation and node type."""
    if node not in G:
        return []

    result = []

    for target in G.successors(node):
        if relation and not relation_exists(
            G,
            node,
            target,
            relation,
        ):
            continue

        if kind and node_type(G, target) != kind:
            continue

        result.append(target)

    return result


# KG traversal
def orders_for_customer(G, customer):
    return outgoing(
        G,
        customer,
        PLACED,
        "order",
    )


def payments_for_order(G, order):
    return outgoing(
        G,
        order,
        HAS_PAYMENT,
        "payment",
    )


def products_for_order(G, order):
    return outgoing(
        G,
        order,
        CONTAINS,
        "product",
    )


def reviews_for_order(G, order):
    return outgoing(
        G,
        order,
        HAS_REVIEW,
        "review",
    )


def categories_for_product(G, product):
    return outgoing(
        G,
        product,
        BELONGS_TO,
        "category",
    )


def sellers_for_product(G, product):
    return outgoing(
        G,
        product,
        OFFERED_BY,
        "seller",
    )


def payment_value(G, payment):
    try:
        return float(
            G.nodes[payment].get("value", 0)
        )
    except (TypeError, ValueError):
        return 0.0


def order_value(G, order):
    return sum(
        payment_value(G, payment)
        for payment in payments_for_order(G, order)
    )


def persistent_customer_id(G, customer):
    return G.nodes[customer].get("unique_id")


def customer_groups(G):
    """Group customer record nodes by persistent customer_unique_id."""
    groups = defaultdict(list)

    for customer in nodes_of_type(G, "customer"):
        uid = persistent_customer_id(G, customer)

        if uid:
            groups[str(uid)].append(customer)

    return groups


# main analytical functions
def customer_spending(G):
    """Calculate historical payment value for each persistent customer."""
    rows = []

    for uid, customers in customer_groups(G).items():
        orders = set()

        for customer in customers:
            orders.update(
                orders_for_customer(G, customer)
            )

        spending = sum(
            order_value(G, order)
            for order in orders
        )
        order_count = len(orders)

        rows.append({
            "customer_unique_id": uid,
            "customer_record_count": len(customers),
            "order_count": order_count,
            "total_spending": spending,
            "average_order_value": (
                spending / order_count
                if order_count
                else 0.0
            ),
        })

    columns = [
        "customer_unique_id",
        "customer_record_count",
        "order_count",
        "total_spending",
        "average_order_value",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        "total_spending",
        ascending=False,
    ).reset_index(drop=True)


def repeat_customers(G):
    """Return persistent customers with more than one order."""
    df = customer_spending(G)
    return df[
        df["order_count"] > 1
    ].reset_index(drop=True)


def total_revenue(G):
    """
    Calculate total payment value represented in the KG.

    This is a dataset-derived payment/revenue proxy.
    """
    return sum(
        payment_value(G, payment)
        for payment in nodes_of_type(G, "payment")
    )


def average_order_value(G):
    """Calculate average payment value per order."""
    orders = nodes_of_type(G, "order")

    if not orders:
        return 0.0

    return sum(
        order_value(G, order)
        for order in orders
    ) / len(orders)


def product_popularity(G):
    """Count distinct order-product associations for each product."""
    counts = defaultdict(int)

    for order in nodes_of_type(G, "order"):
        for product in products_for_order(G, order):
            counts[product] += 1

    rows = [
        {
            "product_id": product,
            "product": display_name(G, product),
            "order_count": count,
        }
        for product, count in counts.items()
    ]

    columns = [
        "product_id",
        "product",
        "order_count",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        "order_count",
        ascending=False,
    ).reset_index(drop=True)


def category_popularity(G):
    """Count associated order-product paths for each category."""
    counts = defaultdict(int)

    for order in nodes_of_type(G, "order"):
        for product in products_for_order(G, order):
            for category in categories_for_product(
                G,
                product,
            ):
                counts[category] += 1

    rows = [
        {
            "category_id": category,
            "category": display_name(G, category),
            "order_count": count,
        }
        for category, count in counts.items()
    ]

    columns = [
        "category_id",
        "category",
        "order_count",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        "order_count",
        ascending=False,
    ).reset_index(drop=True)


def seller_popularity(G):
    """Count product-order associations connected to each seller."""
    counts = defaultdict(int)

    for order in nodes_of_type(G, "order"):
        for product in products_for_order(G, order):
            for seller in sellers_for_product(
                G,
                product,
            ):
                counts[seller] += 1

    rows = [
        {
            "seller_id": seller,
            "seller": display_name(G, seller),
            "product_order_count": count,
        }
        for seller, count in counts.items()
    ]

    columns = [
        "seller_id",
        "seller",
        "product_order_count",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        "product_order_count",
        ascending=False,
    ).reset_index(drop=True)


def sentiment_summary(G):
    """Count review sentiment nodes derived from the KG."""
    counts = defaultdict(int)

    for review in nodes_of_type(G, "review"):
        sentiments = outgoing(
            G,
            review,
            HAS_SENTIMENT,
            "sentiment",
        )

        for sentiment in sentiments:
            counts[
                display_name(G, sentiment)
            ] += 1

    rows = [
        {
            "sentiment": sentiment,
            "review_count": count,
        }
        for sentiment, count in counts.items()
    ]

    columns = [
        "sentiment",
        "review_count",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        "review_count",
        ascending=False,
    ).reset_index(drop=True)


# customer-specific traversal
def customer_products(G, customer):
    products = set()

    for order in orders_for_customer(G, customer):
        products.update(
            products_for_order(G, order)
        )

    return products


def customer_categories(G, customer):
    categories = set()

    for product in customer_products(G, customer):
        categories.update(
            categories_for_product(G, product)
        )

    return categories


def customer_sellers(G, customer):
    sellers = set()

    for product in customer_products(G, customer):
        sellers.update(
            sellers_for_product(G, product)
        )

    return sellers


def customer_sentiments(G, customer):
    sentiments = set()

    for order in orders_for_customer(G, customer):
        for review in reviews_for_order(G, order):
            sentiments.update(
                outgoing(
                    G,
                    review,
                    HAS_SENTIMENT,
                    "sentiment",
                )
            )

    return sentiments


# deterministic natural-language intent detection
INTENTS = {
    "count_customers": [
        "how many customers",
        "number of customers",
        "customer count",
        "count customers",
        "customers in the kg",
        "customers are in the kg",
        "total customers",
    ],
    "count_orders": [
        "how many orders",
        "number of orders",
        "order count",
        "count orders",
        "orders in the kg",
        "total orders",
    ],
    "count_products": [
        "how many products",
        "number of products",
        "product count",
        "count products",
        "products in the kg",
        "total products",
    ],
    "count_categories": [
        "how many categories",
        "number of categories",
        "category count",
        "count categories",
        "categories in the kg",
        "total categories",
    ],
    "count_sellers": [
        "how many sellers",
        "number of sellers",
        "seller count",
        "count sellers",
        "sellers in the kg",
        "total sellers",
    ],
    "count_payments": [
        "how many payments",
        "number of payments",
        "payment count",
        "count payments",
        "payments in the kg",
        "total payments",
    ],
    "count_reviews": [
        "how many reviews",
        "number of reviews",
        "review count",
        "count reviews",
        "reviews in the kg",
        "total reviews",
    ],
    "repeat_customers": [
        "repeat customer",
        "repeat customers",
        "returning customer",
        "returning customers",
        "more than one order",
        "more than one purchase",
        "multiple orders",
        "multiple purchases",
        "ordered more than once",
        "ordered multiple times",
    ],
    "total_revenue": [
        "total revenue",
        "overall revenue",
        "sales revenue",
        "total sales",
        "total payment value",
        "total payment",
        "payment value",
        "overall payment",
        "how much revenue",
        "how much did we make",
        "how much was paid",
    ],
    "average_order_value": [
        "average order value",
        "average order",
        "aov",
        "average spending per order",
        "average payment value",
        "average payment",
    ],
    "product_popularity": [
        "popular product",
        "popular products",
        "top product",
        "top products",
        "best selling product",
        "best selling products",
        "most purchased product",
        "most purchased products",
        "most popular product",
        "most popular products",
    ],
    "category_popularity": [
        "popular category",
        "popular categories",
        "top category",
        "top categories",
        "top product category",
        "top product categories",
        "best category",
        "best categories",
        "most popular category",
        "most popular categories",
        "which categories have the most orders",
        "which product categories have the most orders",
    ],
    "seller_popularity": [
        "top seller",
        "top sellers",
        "popular seller",
        "popular sellers",
        "best seller",
        "best sellers",
        "seller performance",
        "most popular seller",
        "most popular sellers",
    ],
    "customer_spending": [
        "customer spending",
        "customers by spending",
        "highest spending customer",
        "highest spending customers",
        "top spending customers",
        "top customers",
        "highest value customer",
        "highest value customers",
        "customer value",
        "who spent the most",
        "which customers spent the most",
    ],
    "sentiment": [
        "review sentiment",
        "review sentiments",
        "customer sentiment",
        "sentiment distribution",
        "sentiment",
        "positive reviews",
        "negative reviews",
        "neutral reviews",
    ],
    "customer_products": [
        "what products",
        "which products",
        "products purchased",
        "products bought",
        "products did",
    ],
    "customer_categories": [
        "what categories",
        "which categories",
        "categories purchased",
        "categories bought",
    ],
    "customer_sellers": [
        "which sellers",
        "what sellers",
        "seller purchased",
        "purchased from seller",
        "bought from seller",
    ],
}


def detect_intent(question):
    q = normalize(question)

    # category detection must precede product detection because
    # phrases such as "top product categories" contain "top product".
    priority = [
        "count_customers",
        "count_orders",
        "count_products",
        "count_categories",
        "count_sellers",
        "count_payments",
        "count_reviews",
        "category_popularity",
        "total_revenue",
        "average_order_value",
        "repeat_customers",
        "customer_spending",
        "seller_popularity",
        "sentiment",
        "customer_products",
        "customer_categories",
        "customer_sellers",
        "product_popularity",
    ]

    for intent in priority:
        if has_any(q, INTENTS[intent]):
            return intent

    return "unknown"


def extract_top_n(question, default=10):
    """Extract an explicit top-N request, otherwise use the default."""
    q = normalize(question)

    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bshow\s+(\d+)\s+(?:repeat|top|highest|most|best|popular)",
        r"\b(\d+)\s+(?:repeat\s+customers?|customers?|products?|categories?|sellers?)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            value = int(match.group(1))
            return max(
                1,
                min(value, 1000),
            )

    return default


# customer ID resolution
def extract_customer_id(question):
    match = re.search(
        r"\b[a-fA-F0-9]{32}\b",
        str(question),
    )
    return (
        match.group(0).lower()
        if match
        else None
    )


def resolve_customer(G, question):
    identifier = extract_customer_id(question)

    if not identifier:
        return None

    for customer in nodes_of_type(G, "customer"):
        if str(customer).lower() == identifier:
            return customer

        uid = persistent_customer_id(
            G,
            customer,
        )

        if uid and str(uid).lower() == identifier:
            return customer

    return None


# customer summary
def customer_summary(G, customer):
    orders = orders_for_customer(
        G,
        customer,
    )
    products = customer_products(
        G,
        customer,
    )
    categories = customer_categories(
        G,
        customer,
    )
    sellers = customer_sellers(
        G,
        customer,
    )
    spending = sum(
        order_value(G, order)
        for order in orders
    )

    return {
        "customer_id": str(customer),
        "customer_unique_id": persistent_customer_id(
            G,
            customer,
        ),
        "order_count": len(orders),
        "product_count": len(products),
        "category_count": len(categories),
        "seller_count": len(sellers),
        "total_spending": spending,
        "average_order_value": (
            spending / len(orders)
            if orders
            else 0.0
        ),
    }


# natural-language KG service
def ask_knowledge_graph(G, question, top_n=None):
    """
    Answer a supported natural-language business question.

    The returned dictionary always contains:
        question, intent, answer, data, paths, supported
    """
    question = str(question).strip()

    if top_n is None:
        top_n = extract_top_n(question)

    if not question:
        return {
            "question": question,
            "intent": "unknown",
            "answer": (
                "Please enter a Knowledge Graph question."
            ),
            "data": None,
            "paths": [],
            "supported": False,
        }

    intent = detect_intent(question)

    # resolve the customer before global sentiment handling so that
    # customer-specific sentiment queries are not treated as global ones
    customer = resolve_customer(
        G,
        question,
    )

    # customer-specific questions
    if customer is not None:
        if intent == "customer_products":
            products = customer_products(
                G,
                customer,
            )

            data = pd.DataFrame([
                {
                    "product_id": p,
                    "product": display_name(G, p),
                }
                for p in products
            ])

            return {
                "question": question,
                "intent": intent,
                "answer": (
                    f"The customer is connected to "
                    f"{len(products):,} purchased products."
                ),
                "data": data.head(top_n),
                "paths": [
                    "Customer -> PLACED -> Order",
                    "Order -> CONTAINS -> Product",
                ],
                "supported": True,
            }

        if intent == "customer_categories":
            categories = customer_categories(
                G,
                customer,
            )

            data = pd.DataFrame([
                {
                    "category_id": c,
                    "category": display_name(G, c),
                }
                for c in categories
            ])

            return {
                "question": question,
                "intent": intent,
                "answer": (
                    f"The customer is connected to "
                    f"{len(categories):,} purchased categories."
                ),
                "data": data.head(top_n),
                "paths": [
                    "Customer -> PLACED -> Order",
                    "Order -> CONTAINS -> Product",
                    "Product -> BELONGS_TO -> Category",
                ],
                "supported": True,
            }

        if intent == "customer_sellers":
            sellers = customer_sellers(
                G,
                customer,
            )

            data = pd.DataFrame([
                {
                    "seller_id": s,
                    "seller": display_name(G, s),
                }
                for s in sellers
            ])

            return {
                "question": question,
                "intent": intent,
                "answer": (
                    f"The customer is connected to "
                    f"{len(sellers):,} sellers."
                ),
                "data": data.head(top_n),
                "paths": [
                    "Customer -> PLACED -> Order",
                    "Order -> CONTAINS -> Product",
                    "Product -> OFFERED_BY -> Seller",
                ],
                "supported": True,
            }

        if intent == "sentiment":
            sentiments = customer_sentiments(
                G,
                customer,
            )

            data = pd.DataFrame([
                {
                    "sentiment": display_name(
                        G,
                        sentiment,
                    )
                }
                for sentiment in sentiments
            ])

            return {
                "question": question,
                "intent": "customer_sentiment",
                "answer": (
                    f"Found {len(sentiments)} sentiment node(s) "
                    "connected to this customer's reviews."
                ),
                "data": data,
                "paths": [
                    "Customer -> PLACED -> Order",
                    "Order -> HAS_REVIEW -> Review",
                    "Review -> HAS_SENTIMENT -> Sentiment",
                ],
                "supported": True,
            }

        summary = customer_summary(
            G,
            customer,
        )

        return {
            "question": question,
            "intent": "customer_summary",
            "answer": (
                f"This customer has "
                f"{summary['order_count']} order(s), "
                f"€{summary['total_spending']:,.2f} in represented "
                "payment value, and "
                f"{summary['product_count']} purchased product(s)."
            ),
            "data": pd.DataFrame([summary]),
            "paths": [
                "Customer -> PLACED -> Order",
                "Order -> HAS_PAYMENT -> Payment",
                "Order -> CONTAINS -> Product",
            ],
            "supported": True,
        }

    # global count questions
    count_types = {
        "count_customers": (
            "customer",
            "customers",
        ),
        "count_orders": (
            "order",
            "orders",
        ),
        "count_products": (
            "product",
            "products",
        ),
        "count_categories": (
            "category",
            "categories",
        ),
        "count_sellers": (
            "seller",
            "sellers",
        ),
        "count_payments": (
            "payment",
            "payments",
        ),
        "count_reviews": (
            "review",
            "reviews",
        ),
    }

    if intent in count_types:
        entity_type, label = count_types[intent]
        count = len(
            nodes_of_type(
                G,
                entity_type,
            )
        )

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"The KG contains {count:,} {label}."
            ),
            "data": pd.DataFrame([{
                "entity_type": entity_type,
                "count": count,
            }]),
            "paths": [
                f"KG node type: {entity_type}"
            ],
            "supported": True,
        }

    # global questions
    if intent == "total_revenue":
        revenue = total_revenue(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"Total payment value represented in the KG is "
                f"€{revenue:,.2f}."
            ),
            "data": pd.DataFrame([{
                "total_payment_value": revenue
            }]),
            "paths": [
                "Order -> HAS_PAYMENT -> Payment"
            ],
            "supported": True,
        }

    if intent == "average_order_value":
        aov = average_order_value(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"The average order payment value is "
                f"€{aov:,.2f}."
            ),
            "data": pd.DataFrame([{
                "average_order_value": aov
            }]),
            "paths": [
                "Order -> HAS_PAYMENT -> Payment"
            ],
            "supported": True,
        }

    if intent == "repeat_customers":
        df = repeat_customers(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"The KG contains {len(df):,} persistent customers "
                "with more than one order. "
                f"Showing the top {min(top_n, len(df))}."
            ),
            "data": df.head(top_n),
            "paths": [
                "Customer -> PLACED -> Order",
                "Customer.unique_id -> persistent customer identity",
            ],
            "supported": True,
        }

    if intent == "customer_spending":
        df = customer_spending(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"Found {len(df):,} persistent customers. "
                f"Showing the top {min(top_n, len(df))} by "
                "historical payment value."
            ),
            "data": df.head(top_n),
            "paths": [
                "Customer -> PLACED -> Order",
                "Order -> HAS_PAYMENT -> Payment",
                "Customer.unique_id -> persistent identity",
            ],
            "supported": True,
        }

    if intent == "category_popularity":
        df = category_popularity(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"Showing the top {min(top_n, len(df))} categories "
                "by associated order-product paths."
            ),
            "data": df.head(top_n),
            "paths": [
                "Order -> CONTAINS -> Product",
                "Product -> BELONGS_TO -> Category",
            ],
            "supported": True,
        }

    if intent == "product_popularity":
        df = product_popularity(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"Showing the top {min(top_n, len(df))} products "
                "by number of associated orders."
            ),
            "data": df.head(top_n),
            "paths": [
                "Order -> CONTAINS -> Product"
            ],
            "supported": True,
        }

    if intent == "seller_popularity":
        df = seller_popularity(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                f"Showing the top {min(top_n, len(df))} sellers "
                "by product-order associations."
            ),
            "data": df.head(top_n),
            "paths": [
                "Order -> CONTAINS -> Product",
                "Product -> OFFERED_BY -> Seller",
            ],
            "supported": True,
        }

    if intent == "sentiment":
        df = sentiment_summary(G)

        return {
            "question": question,
            "intent": intent,
            "answer": (
                "Showing review sentiment counts represented in the KG."
            ),
            "data": df,
            "paths": [
                "Order -> HAS_REVIEW -> Review",
                "Review -> HAS_SENTIMENT -> Sentiment",
            ],
            "supported": True,
        }

    # unsupported question
    return {
        "question": question,
        "intent": "unknown",
        "answer": (
            "I cannot answer that question from the current "
            "Knowledge Graph. Try asking about entity counts, "
            "payment value, order value, repeat customers, "
            "customer spending, products, categories, sellers, "
            "reviews, or customer-specific information."
        ),
        "data": None,
        "paths": [],
        "supported": False,
    }


# pipeline check
def run_graph_queries(G, interactive=False):
    """Run lightweight validation used by main.py."""
    required = [
        PLACED,
        HAS_PAYMENT,
        CONTAINS,
        OFFERED_BY,
        BELONGS_TO,
        HAS_REVIEW,
        HAS_SENTIMENT,
        LOCATED_IN,
    ]

    counts = defaultdict(int)

    for source, target in G.edges():
        for relation in required:
            if relation_exists(
                G,
                source,
                target,
                relation,
            ):
                counts[relation] += 1

    repeats = repeat_customers(G)

    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH QUERY SERVICE")
    print("=" * 70)
    print(
        f"Graph nodes: {G.number_of_nodes():,}"
    )
    print(
        f"Graph edges: {G.number_of_edges():,}"
    )

    print("\nRequired relationships:")

    for relation in required:
        print(
            f"  {relation:<20} {counts[relation]:,}"
        )

    print(
        f"\nPersistent repeat customers: "
        f"{len(repeats):,}"
    )
    print(
        "Natural-language query service: READY"
    )
    print("=" * 70)

    return {
        "ready": True,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "repeat_customers": len(repeats),
        "relationship_counts": dict(counts),
    }