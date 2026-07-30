import networkx as nx
import pandas as pd



def build_graph(
    customers,
    orders,
    order_items,
    products,
    payments,
    sellers,
    reviews,
    geo,
    category
):


    G = nx.DiGraph()



    # ==========================================================
    # KNOWLEDGE GRAPH SCHEMA
    # ==========================================================


    SCHEMA = {


        "customer":
        "entity",


        "order":
        "event",


        "product":
        "entity",


        "seller":
        "entity",


        "payment":
        "financial_event",


        "category":
        "concept",


        "city":
        "location",


        "state":
        "location",


        "review":
        "feedback_event",


        "sentiment":
        "concept"

    }



    # ==========================================================
    # CUSTOMER NODES
    # ==========================================================


    for _, row in customers.iterrows():


        customer_id = row["customer_id"]


        G.add_node(

            customer_id,

            type="customer",

            schema_type=SCHEMA["customer"],

            display_name=
            f"Customer | {row['customer_city']} | {row['customer_state']}",

            unique_id=
            row["customer_unique_id"],

            city=
            row["customer_city"],

            state=
            row["customer_state"]

        )



        # -------------------------
        # City node
        # -------------------------


        city_node = (

            "CITY_"
            +
            str(row["customer_city"])

        )


        G.add_node(

            city_node,

            type="city",

            schema_type=SCHEMA["city"],

            display_name=
            f"City | {row['customer_city']}"

        )


        G.add_edge(

            customer_id,

            city_node,

            relation="LOCATED_IN"

        )



        # -------------------------
        # State node
        # -------------------------


        state_node = (

            "STATE_"
            +
            str(row["customer_state"])

        )


        G.add_node(

            state_node,

            type="state",

            schema_type=SCHEMA["state"],

            display_name=
            f"State | {row['customer_state']}"

        )


        G.add_edge(

            city_node,

            state_node,

            relation="LOCATED_IN"

        )



    # ==========================================================
    # ORDER NODES
    # ==========================================================


    for _, row in orders.iterrows():


        order_id = row["order_id"]



        G.add_node(

            order_id,

            type="order",

            schema_type=SCHEMA["order"],

            display_name=
            f"Order | {order_id[:8]}",

            status=
            row["order_status"],

            purchase_date=
            str(row["order_purchase_timestamp"])

        )



        if row["customer_id"] in G:


            G.add_edge(

                row["customer_id"],

                order_id,

                relation="PLACED"

            )



    # ==========================================================
    # SELLER NODES
    # ==========================================================


    for _, row in sellers.iterrows():


        seller_id=row["seller_id"]



        G.add_node(

            seller_id,

            type="seller",

            schema_type=SCHEMA["seller"],

            display_name=
            f"Seller | {row['seller_city']} | {row['seller_state']}",

            city=
            row["seller_city"],

            state=
            row["seller_state"]

        )



        city_node = (

            "SELLER_CITY_"

            +

            str(row["seller_city"])

        )



        G.add_node(

            city_node,

            type="city",

            schema_type=SCHEMA["city"],

            display_name=
            f"City | {row['seller_city']}"

        )



        G.add_edge(

            seller_id,

            city_node,

            relation="LOCATED_IN"

        )



        state_node = (

            "STATE_"

            +

            str(row["seller_state"])

        )


        G.add_edge(

            city_node,

            state_node,

            relation="LOCATED_IN"

        )



    # ==========================================================
    # PRODUCTS
    # ==========================================================


    for _, row in products.iterrows():


        product_id=row["product_id"]



        G.add_node(

            product_id,

            type="product",

            schema_type=SCHEMA["product"],

            display_name=
            f"Product | {product_id[:8]}",

            weight=
            row["product_weight_g"],

            length=
            row["product_length_cm"],

            height=
            row["product_height_cm"],

            width=
            row["product_width_cm"]

        )



        if pd.notna(

            row.get(
                "product_category_name_english"
            )

        ):


            english_category = (

                row["product_category_name_english"]

            )


            original_category = (

                row["product_category_name"]

            )



            # English category node


            G.add_node(

                english_category,

                type="category",

                schema_type=SCHEMA["category"],

                display_name=
                f"Category | {english_category}"

            )



            G.add_edge(

                product_id,

                english_category,

                relation="BELONGS_TO"

            )



            # Original Portuguese category


            G.add_node(

                original_category,

                type="category_original",

                display_name=
                f"Original Category | {original_category}"

            )



            G.add_edge(

                english_category,

                original_category,

                relation="TRANSLATED_FROM"

            )



    # ==========================================================
    # ORDER ITEMS
    # ==========================================================


    for _, row in order_items.iterrows():


        order_id=row["order_id"]

        product_id=row["product_id"]

        seller_id=row["seller_id"]



        if order_id in G and product_id in G:


            G.add_edge(

                order_id,

                product_id,

                relation="CONTAINS"

            )



        if order_id in G and seller_id in G:


            G.add_edge(

                order_id,

                seller_id,

                relation="SOLD_BY"

            )



        if product_id in G and seller_id in G:


            G.add_edge(

                product_id,

                seller_id,

                relation="OFFERED_BY"

            )



    # ==========================================================
    # PAYMENTS
    # ==========================================================


    for idx,row in payments.iterrows():


        payment_node = (

            "PAYMENT_"

            +

            str(idx)

        )


        G.add_node(

            payment_node,

            type="payment",

            schema_type=SCHEMA["payment"],

            display_name=
            f"Payment | {row['payment_type']}",

            payment_type=
            row["payment_type"],

            installments=
            row["payment_installments"],

            value=
            row["payment_value"]

        )



        if row["order_id"] in G:


            G.add_edge(

                row["order_id"],

                payment_node,

                relation="HAS_PAYMENT"

            )



    # ==========================================================
    # REVIEWS
    # ==========================================================


    for idx,row in reviews.iterrows():


        review_node = (

            "REVIEW_"

            +

            str(idx)

        )



        G.add_node(

            review_node,

            type="review",

            schema_type=SCHEMA["review"],

            display_name=
            f"Review | score {row['review_score']}",

            score=
            row["review_score"],

            comment=
            str(row.get("review_comment_message",""))

        )



        if row["order_id"] in G:


            G.add_edge(

                row["order_id"],

                review_node,

                relation="HAS_REVIEW"

            )



        sentiment = "neutral"



        if row["review_score"] >=4:

            sentiment="positive"


        elif row["review_score"] <=2:

            sentiment="negative"



        sentiment_node = (

            "SENTIMENT_"

            +

            sentiment

        )



        G.add_node(

            sentiment_node,

            type="sentiment",

            display_name=
            f"Sentiment | {sentiment}"

        )



        G.add_edge(

            review_node,

            sentiment_node,

            relation="HAS_SENTIMENT"

        )



    # ==========================================================
    # GRAPH SUMMARY
    # ==========================================================


    print("\n--- KNOWLEDGE GRAPH CREATED ---")


    print(
        "Nodes :",
        G.number_of_nodes()
    )


    print(
        "Edges :",
        G.number_of_edges()
    )



    print("\nNode Types")



    node_types={}



    for _,data in G.nodes(data=True):


        t=data.get(
            "type",
            "unknown"
        )


        node_types[t]=node_types.get(t,0)+1



    for t,c in sorted(node_types.items()):


        print(
            f"{t:<20}: {c}"
        )



    return G