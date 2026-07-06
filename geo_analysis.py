from sklearn.cluster import KMeans
import pandas as pd


def run_geo_analysis(geo):

    print("\n--- GEOSPATIAL KG SUBGRAPH DISCOVERY ---")

    geo_clean = geo.dropna(subset=["geolocation_lat", "geolocation_lng"]).copy()

    coords = geo_clean[["geolocation_lat", "geolocation_lng"]]

    # -------------------------
    # KG CLUSTERING = STRUCTURAL PARTITIONING
    # -------------------------
    kmeans = KMeans(n_clusters=8, n_init=10, random_state=42)
    geo_clean["cluster"] = kmeans.fit_predict(coords)

    print("\nDERIVED FACT: Geographic Clusters")
    print(geo_clean["cluster"].value_counts().head())

    print("\n--- INTERPRETATION ---")
    print("Clusters represent latent regional subgraphs in the KG")
    print("Useful for supply-chain / demand partitioning")