import kagglehub
import pandas as pd
import os

def load_data():
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

    print("Dataset path:", path)
    print("Files:", os.listdir(path))

    customers = pd.read_csv(f"{path}/olist_customers_dataset.csv")
    orders = pd.read_csv(f"{path}/olist_orders_dataset.csv")
    order_items = pd.read_csv(f"{path}/olist_order_items_dataset.csv")
    products = pd.read_csv(f"{path}/olist_products_dataset.csv")
    payments = pd.read_csv(f"{path}/olist_order_payments_dataset.csv")
    sellers = pd.read_csv(f"{path}/olist_sellers_dataset.csv")
    reviews = pd.read_csv(f"{path}/olist_order_reviews_dataset.csv")
    geo = pd.read_csv(f"{path}/olist_geolocation_dataset.csv")
    category = pd.read_csv(f"{path}/product_category_name_translation.csv")

    return customers, orders, order_items, products, payments, sellers, reviews, geo, category