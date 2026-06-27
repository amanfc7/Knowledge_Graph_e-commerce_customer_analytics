from sklearn.cluster import KMeans
import pandas as pd

def run_geo_analysis(geo_df):

    print("\n--- GEO CLUSTERING (EXCEED LO) ---")

    coords = geo_df[["geolocation_lat", "geolocation_lng"]].dropna()

    kmeans = KMeans(n_clusters=10, n_init=10)
    geo_df["cluster"] = kmeans.fit_predict(coords)

    print("\nCluster counts:")
    print(geo_df["cluster"].value_counts().head())