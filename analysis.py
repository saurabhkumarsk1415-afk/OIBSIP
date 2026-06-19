"""
Marketing Analytics: Customer Segmentation Analysis
Project 2 Proposal - Level 1
Dataset: https://www.kaggle.com/code/analystoleksandra/marketing-analytics-customer-segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv('marketing_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ── 2. Feature Engineering ───────────────────────────────────────────────
df['Age']              = 2024 - df['Year_Birth']
df['TotalSpend']       = df[['MntWines','MntFruits','MntMeatProducts',
                              'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
df['TotalPurchases']   = df[['NumDealsPurchases','NumWebPurchases',
                              'NumCatalogPurchases','NumStorePurchases']].sum(axis=1)
df['TotalCampaigns']   = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                              'AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
df['Children']         = df['Kidhome'] + df['Teenhome']
df['SpendPerPurchase'] = df['TotalSpend'] / (df['TotalPurchases'] + 1)

# Impute missing income with median
df['Income'] = df['Income'].fillna(df['Income'].median())

# Remove outliers
df = df[(df['Age'] < 90) & (df['Income'] < 160000)]

# ── 3. Descriptive Statistics ────────────────────────────────────────────
print("\n=== Descriptive Statistics ===")
print(df[['Age','Income','TotalSpend','TotalPurchases']].describe().round(2))

# ── 4. K-Means Clustering ────────────────────────────────────────────────
features = ['Age','Income','TotalSpend','TotalPurchases',
            'Recency','Children','NumWebVisitsMonth']
X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=20)
df.loc[X.index, 'Cluster'] = km.fit_predict(X_scaled)

# PCA for 2D visualisation
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ── 5. Cluster Profiles ──────────────────────────────────────────────────
profile = df[df['Cluster'].notna()].groupby('Cluster')[
    ['Age','Income','TotalSpend','TotalPurchases','Children','TotalCampaigns']
].mean().round(1)
profile['Count'] = df[df['Cluster'].notna()].groupby('Cluster').size()
print("\n=== Cluster Profiles ===")
print(profile)

print("\nAnalysis complete. Run the full script to generate all figures.")
