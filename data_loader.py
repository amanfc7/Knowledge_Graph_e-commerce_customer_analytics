import kagglehub
import pandas as pd
import os


def load_data():

    print("\n--- LOADING KG DATA LAYER (GROUND FACTS) ---")

    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

    print("Dataset path:", path)
    print("Files found:", os.listdir(path))

    # -------------------------
    # CORE ENTITIES
    # -------------------------
    customers = pd.read_csv(f"{path}/olist_customers_dataset.csv")
    orders = pd.read_csv(f"{path}/olist_orders_dataset.csv")
    order_items = pd.read_csv(f"{path}/olist_order_items_dataset.csv")
    products = pd.read_csv(f"{path}/olist_products_dataset.csv")
    payments = pd.read_csv(f"{path}/olist_order_payments_dataset.csv")

    # -------------------------
    # OPTIONAL EXTENSION LAYERS
    # -------------------------
    sellers = pd.read_csv(f"{path}/olist_sellers_dataset.csv")
    reviews = pd.read_csv(f"{path}/olist_order_reviews_dataset.csv")
    geo = pd.read_csv(f"{path}/olist_geolocation_dataset.csv")
    category = pd.read_csv(f"{path}/product_category_name_translation.csv")

    # -------------------------
    # BASIC DATA VALIDATION (KGMS STYLE)
    # -------------------------
    print("\n--- DATA VALIDATION ---")
    print("Customers:", len(customers))
    print("Orders:", len(orders))
    print("Products:", len(products))
    print("Payments:", len(payments))

    # Remove obvious null-heavy rows
    orders = orders.dropna(subset=["order_id", "customer_id"])
    payments = payments.dropna(subset=["order_id", "payment_value"])

    return customers, orders, order_items, products, payments, sellers, reviews, geo, category