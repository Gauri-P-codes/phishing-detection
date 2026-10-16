"""Unit V - Unsupervised learning on the phishing feature space:
K-means (+elbow/silhouette), DBSCAN (outliers), Apriori (frequent patterns) & association rules."""
from itertools import combinations
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def kmeans_study(Xs, y, ks=range(2, 9)):
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=0).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_, sample_size=2000, random_state=0)})
    res = pd.DataFrame(rows)
    km = KMeans(2, n_init=10, random_state=0).fit(Xs)
    purity = sum(np.bincount(y[km.labels_ == c]).max() for c in set(km.labels_)) / len(y)
    return res, km, purity

def dbscan_outliers(Xs, eps=2.5, min_samples=10):
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    return lab, float((lab == -1).mean())

def apriori(transactions, min_support=0.15, max_len=3):
    """transactions: list of frozensets. Returns {itemset: support}. Classic level-wise algorithm."""
    n = len(transactions)
    count = lambda c: sum(c <= t for t in transactions) / n
    items = {i for t in transactions for i in t}
    L = {frozenset([i]): count(frozenset([i])) for i in items}
    L = {s: v for s, v in L.items() if v >= min_support}; freq = dict(L); k = 2
    while L and k <= max_len:
        cands = {a | b for a in L for b in L if len(a | b) == k}
        cands = {c for c in cands if all(frozenset(s) in L for s in combinations(c, k - 1))}   # Apriori pruning
        L = {c: count(c) for c in cands}; L = {s: v for s, v in L.items() if v >= min_support}
        freq.update(L); k += 1
    return freq

def rules(freq, min_conf=0.8):
    out = []
    for s, sup in freq.items():
        if len(s) < 2: continue
        for r in range(1, len(s)):
            for a in map(frozenset, combinations(s, r)):
                c = s - a; conf = sup / freq[a]; lift = conf / freq[c]
                if conf >= min_conf: out.append((set(a), set(c), sup, conf, lift))
    return sorted(out, key=lambda r: (-r[3], -r[4]))

def to_transactions(feat_df, y):
    """Binary-ise features into items like 'has_ip', 'url_len=HIGH', plus class item 'PHISHING'/'LEGIT'."""
    T = []
    bin_cols = [c for c in feat_df.columns if set(feat_df[c].unique()) <= {0, 1}]
    num_cols = ["url_len", "n_subdomains", "host_entropy", "path_depth"]
    thr = {c: feat_df[c].median() for c in num_cols}
    for (_, r), lab in zip(feat_df.iterrows(), y):
        s = {c for c in bin_cols if r[c] == 1}
        s |= {f"{c}=HIGH" for c in num_cols if r[c] > thr[c]}
        s.add("PHISHING" if lab == 1 else "LEGIT"); T.append(frozenset(s))
    return T
