"""Unit III - Data preprocessing: cleaning, scaling, normalisation, discretisation, PCA, stratified sampling."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def clean(df):
    n0 = len(df)
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df.drop_duplicates(subset="url")
    print(f"[clean] {n0} -> {len(df)} rows (nulls/duplicates removed)")
    return df

def split(X, y, test=0.2, seed=42):
    """Stratified sampling keeps the phishing/legit ratio in train and test."""
    return train_test_split(X, y, test_size=test, stratify=y, random_state=seed)

def scale(X_train, X_test, kind="standard"):
    s = StandardScaler() if kind == "standard" else MinMaxScaler()
    return s.fit_transform(X_train), s.transform(X_test), s

def discretize(X, n_bins=5):
    """Equal-width binning (data discretisation) - also feeds Apriori."""
    return KBinsDiscretizer(n_bins, encode="ordinal", strategy="uniform").fit_transform(X)

def pca_reduce(X, var=0.95):
    p = PCA(n_components=var, random_state=0).fit(X)
    print(f"[PCA] {X.shape[1]} features -> {p.n_components_} components ({var:.0%} variance)")
    return p
