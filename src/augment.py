import os, random, shutil
import pandas as pd

random.seed(42)
if not os.path.exists("data/urls_base.csv"):          # backup so re-running never compounds
    shutil.copy("data/urls.csv", "data/urls_base.csv")
base = pd.read_csv("data/urls_base.csv")

PATHS = ["", "", "/about", "/contact", "/help", "/blog", "/careers", "/products", "/news"]

def variant(u):
    return u.replace("://www.", "://").rstrip("/") + random.choice(PATHS)

parts = [base]
for label in (0, 1):                                    # same transformation for both classes
    s = base[base.label == label].sample(3000, random_state=42)
    parts.append(pd.DataFrame({"url": s.url.map(variant), "label": label}))

if os.path.exists("data/extra_legit.txt"):              # optional: your own URLs, one per line
    urls = [l.strip() for l in open("data/extra_legit.txt") if l.strip()]
    parts.append(pd.DataFrame({"url": urls, "label": 0}))

df = pd.concat(parts).drop_duplicates("url").sample(frac=1, random_state=42)
df.to_csv("data/urls.csv", index=False)
print(f"Saved {len(df)} rows | phishing ratio = {df['label'].mean():.2f}")