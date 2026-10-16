"""Unit III - Data transformation / feature engineering: URL -> numeric vector (lexical only, no network)."""
import math, re
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
from .trie import KEYWORD_TRIE, BRAND_TRIE

BAD_TLD = {"xyz","top","tk","ml","ga","cf","click","work","info","buzz"}
SHORTENERS = {"bit.ly","tinyurl.com","goo.gl","t.co"}
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

def entropy(s):
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())

def extract(url: str) -> dict:
    u = url.strip()
    u = re.sub(r"^(https?://)www\.", r"\1", u, flags=re.I)
    p = urlparse(u if "://" in u else "http://" + u)
    host, path, query = (p.hostname or ""), p.path, p.query
    parts = host.split(".")
    tld = parts[-1] if parts else ""
    sld = parts[-2] if len(parts) >= 2 else host
    nd = sum(c.isdigit() for c in u)
    return {
        "url_len": len(u), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        "n_dots": u.count("."), "n_hyphens_host": host.count("-"), "n_at": u.count("@"),
        "n_digits": nd, "digit_ratio": nd / max(len(u), 1),
        "n_subdomains": max(len(parts) - 2, 0), "path_depth": path.count("/"),
        "n_params": query.count("="), "has_ip": int(bool(IP_RE.match(host))),
        "is_https": int(p.scheme == "https"), "bad_tld": int(tld in BAD_TLD),
        "is_shortener": int(host in SHORTENERS), "has_userinfo": int(p.username is not None),
        "double_slash_path": int("//" in path), "host_entropy": entropy(host), "url_entropy": entropy(u),
        "kw_in_url": int(KEYWORD_TRIE.find_in_text(u.lower())),
        "brand_in_subdomain": int(any(BRAND_TRIE.find_in_text(x) for x in parts[:-2])),
        "brand_in_path": int(BRAND_TRIE.find_in_text(path.lower())),
        "brand_not_sld": int(BRAND_TRIE.find_in_text(host) and not BRAND_TRIE.find_in_text(sld)),
    }

def featurize(urls) -> pd.DataFrame:
    return pd.DataFrame([extract(u) for u in urls])
