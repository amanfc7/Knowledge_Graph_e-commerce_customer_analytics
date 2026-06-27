# Multi-Layer E-commerce Knowledge Graph for Customer Behavior and Financial Analysis
 
## Overview
This project builds a multi-layer Knowledge Graph from the Olist Brazilian E-commerce dataset to analyze customer behavior, product relationships, seller performance, and financial transactions. The system integrates graph-based modeling, machine learning embeddings, sentiment analysis, and geospatial clustering to extract meaningful business insights from raw e-commerce data.

The goal is to demonstrate how Knowledge Graphs can unify heterogeneous datasets into a single analytical structure for advanced data science and financial analysis.

---

## Features
- Multi-layer Knowledge Graph construction from 9 interconnected datasets
- Customer behavior analysis (purchase patterns and CLV)
- Financial analytics (total revenue and customer lifetime value)
- Product and category relationship analysis
- Seller performance analysis (revenue and diversity)
- Review sentiment analysis using NLP (TextBlob)
- Geospatial clustering of customer/geolocation data
- Graph embeddings using Node2Vec for similarity analysis
- Graph visualization using NetworkX
- JSON export of Knowledge Graph for interoperability

---

## Dataset
Olist Brazilian E-commerce dataset (Kaggle)

This dataset includes:
- Customers
- Orders
- Order items
- Payments
- Products
- Sellers
- Reviews
- Geolocation data
- Product category translation

It represents a real-world e-commerce ecosystem with financial and behavioral relationships.

---

## Technologies
- Python
- Pandas
- NetworkX
- Matplotlib
- Scikit-learn
- TextBlob (NLP sentiment analysis)
- Node2Vec (graph embeddings)
- KaggleHub
- JSON (graph export format)

---

## Key Components

### 1. Knowledge Graph Layer
Models relationships between customers, orders, products, sellers, payments, and categories using a directed graph structure.

### 2. Analytics Layer
Performs financial and behavioral analysis including:
- Revenue computation
- Customer Lifetime Value (CLV)
- Product demand analysis
- Category trends

### 3. Machine Learning Layer
Applies Node2Vec embeddings to capture structural similarity between nodes in the graph.

### 4. NLP Layer
Analyzes customer review text to compute sentiment scores and identify positive/negative feedback trends.

### 5. Geospatial Layer
Clusters geolocation data to identify regional patterns in customer distribution.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py