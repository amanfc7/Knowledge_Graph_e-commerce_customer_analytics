from sklearn.cluster import KMeans

def run_geo_analysis(geo):

    print("\n--- GEO CLUSTERING ANALYSIS ---")

    geo_clean = geo.dropna(subset=["geolocation_lat", "geolocation_lng"]).copy()

    coords = geo_clean[["geolocation_lat", "geolocation_lng"]]

    kmeans = KMeans(n_clusters=8, n_init=10, random_state=42)
    geo_clean["cluster"] = kmeans.fit_predict(coords)

    print("\nCluster distribution:")
    print(geo_clean["cluster"].value_counts().head())