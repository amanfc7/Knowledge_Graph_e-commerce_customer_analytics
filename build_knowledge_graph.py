
import networkx as nx
import pandas as pd
import re
import unicodedata


def normalize_location(value):
    """Create a stable identifier for city/state nodes."""
    value = "" if pd.isna(value) else str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def build_graph(
    customers, orders, order_items, products, payments,
    sellers, reviews, geo, category
):
    G = nx.DiGraph()

    # knowledge graph schema
    SCHEMA = {
        "customer": "entity",
        "order": "event",
        "product": "entity",
        "seller": "entity",
        "payment": "financial_event",
        "category": "concept",
        "category_original": "concept",
        "city": "location",
        "state": "location",
        "review": "feedback_event",
        "sentiment": "concept"
    }

    # customer nodes
    for _, row in customers.iterrows():
        customer_id = row["customer_id"]
        city = str(row["customer_city"])
        state = str(row["customer_state"])

        G.add_node(
            customer_id,
            type="customer",
            schema_type=SCHEMA["customer"],
            display_name=f"Customer | {city} | {state}",
            unique_id=row["customer_unique_id"],
            city=city,
            state=state
        )

        # city node
        city_node = f"CITY_{normalize_location(city)}"
        G.add_node(
            city_node,
            type="city",
            schema_type=SCHEMA["city"],
            display_name=f"City | {city}",
            location_name=city
        )
        G.add_edge(customer_id, city_node, relation="LOCATED_IN")

        # state node
        state_node = f"STATE_{normalize_location(state)}"
        G.add_node(
            state_node,
            type="state",
            schema_type=SCHEMA["state"],
            display_name=f"State | {state}",
            location_name=state
        )
        G.add_edge(city_node, state_node, relation="LOCATED_IN")

    # order nodes
    for _, row in orders.iterrows():
        order_id = row["order_id"]

        G.add_node(
            order_id,
            type="order",
            schema_type=SCHEMA["order"],
            display_name=f"Order | {order_id[:8]}",
            status=row["order_status"],
            purchase_date=str(row["order_purchase_timestamp"])
        )

        if row["customer_id"] in G:
            G.add_edge(row["customer_id"], order_id, relation="PLACED")

    # seller nodes
    for _, row in sellers.iterrows():
        seller_id = row["seller_id"]
        city = str(row["seller_city"])
        state = str(row["seller_state"])

        G.add_node(
            seller_id,
            type="seller",
            schema_type=SCHEMA["seller"],
            display_name=f"Seller | {city} | {state}",
            city=city,
            state=state
        )

        # city node
        city_node = f"CITY_{normalize_location(city)}"
        G.add_node(
            city_node,
            type="city",
            schema_type=SCHEMA["city"],
            display_name=f"City | {city}",
            location_name=city
        )
        G.add_edge(seller_id, city_node, relation="LOCATED_IN")

        # state node
        state_node = f"STATE_{normalize_location(state)}"
        G.add_node(
            state_node,
            type="state",
            schema_type=SCHEMA["state"],
            display_name=f"State | {state}",
            location_name=state
        )
        G.add_edge(city_node, state_node, relation="LOCATED_IN")

    # products
    for _, row in products.iterrows():
        product_id = row["product_id"]

        G.add_node(
            product_id,
            type="product",
            schema_type=SCHEMA["product"],
            display_name=f"Product | {product_id[:8]}",
            weight=row["product_weight_g"],
            length=row["product_length_cm"],
            height=row["product_height_cm"],
            width=row["product_width_cm"]
        )

        if pd.notna(row.get("product_category_name_english")):
            english_category = str(row["product_category_name_english"]).strip()
            original_category = str(row["product_category_name"]).strip()

            # english category node
            english_node = f"CATEGORY_{normalize_location(english_category)}"
            G.add_node(
                english_node,
                type="category",
                schema_type=SCHEMA["category"],
                display_name=f"Category | {english_category}",
                category_name=english_category
            )
            G.add_edge(product_id, english_node, relation="BELONGS_TO")

            # original Portuguese category
            original_node = f"CATEGORY_ORIGINAL_{normalize_location(original_category)}"
            G.add_node(
                original_node,
                type="category_original",
                schema_type=SCHEMA["category_original"],
                display_name=f"Original Category | {original_category}",
                category_name=original_category
            )
            G.add_edge(english_node, original_node, relation="TRANSLATED_FROM")

    # order items
    for _, row in order_items.iterrows():
        order_id = row["order_id"]
        product_id = row["product_id"]
        seller_id = row["seller_id"]

        # order contains product
        if order_id in G and product_id in G:
            G.add_edge(order_id, product_id, relation="CONTAINS")

        # product is offered by seller
        # This is the primitive seller relationship.
        # We do not create Order -> Seller because an order
        # can contain products from multiple sellers.
        if product_id in G and seller_id in G:
            G.add_edge(product_id, seller_id, relation="OFFERED_BY")

    # payments
    for idx, row in payments.iterrows():
        payment_node = f"PAYMENT_{idx}"

        G.add_node(
            payment_node,
            type="payment",
            schema_type=SCHEMA["payment"],
            display_name=f"Payment | {row['payment_type']}",
            payment_type=row["payment_type"],
            installments=row["payment_installments"],
            value=row["payment_value"]
        )

        if row["order_id"] in G:
            G.add_edge(row["order_id"], payment_node, relation="HAS_PAYMENT")

    # reviews
    for idx, row in reviews.iterrows():
        review_node = f"REVIEW_{idx}"

        G.add_node(
            review_node,
            type="review",
            schema_type=SCHEMA["review"],
            display_name=f"Review | score {row['review_score']}",
            score=row["review_score"],
            comment=str(row.get("review_comment_message", ""))
        )

        if row["order_id"] in G:
            G.add_edge(row["order_id"], review_node, relation="HAS_REVIEW")

        # rating-based sentiment
        sentiment = "neutral"
        if row["review_score"] >= 4:
            sentiment = "positive"
        elif row["review_score"] <= 2:
            sentiment = "negative"

        sentiment_node = f"SENTIMENT_{sentiment}"
        G.add_node(
            sentiment_node,
            type="sentiment",
            schema_type=SCHEMA["sentiment"],
            display_name=f"Sentiment | {sentiment}",
            source="review_score",
            sentiment_method="rating_based"
        )
        G.add_edge(review_node, sentiment_node, relation="HAS_SENTIMENT")

    # graph summary
    print("\n--- knowledge graph created ---")
    print("Nodes :", G.number_of_nodes())
    print("Edges :", G.number_of_edges())
    print("\nNode Types")

    node_types = {}
    for _, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    for node_type, count in sorted(node_types.items()):
        print(f"{node_type:<20}: {count}")

    return G
