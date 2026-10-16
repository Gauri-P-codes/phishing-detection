"""Synthetic URL dataset generator (so the project runs offline).
For the final submission REPLACE with a real dataset (see README):
  - PhiUSIIL Phishing URL dataset (UCI)  - Kaggle 'Phishing Site URLs'
Any CSV with columns  url,label  (1 = phishing, 0 = legitimate) works.
"""
import random, argparse
import pandas as pd

BRANDS = ["google","paypal","amazon","microsoft","apple","netflix","sbi","hdfcbank","icicibank",
          "facebook","instagram","flipkart","paytm","irctc","github","wikipedia","linkedin","zoom"]
WORDS = ["news","shop","blog","mail","cloud","learn","docs","music","travel","food","tech","sports"]
GOOD_TLD = ["com","org","in","edu","net","co.in","gov.in"]
BAD_TLD = ["xyz","top","tk","ml","ga","cf","click","work","info","buzz"]
KEYS = ["login","secure","verify","update","account","confirm","banking","signin","webscr","password"]
SHORT = ["bit.ly","tinyurl.com","goo.gl","t.co"]

def _rand(rng, n): return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=n))

def legit(rng):
    host = f"{rng.choice(WORDS)}{rng.choice(['','-'+rng.choice(WORDS)])}"
    if rng.random() < .35: host = rng.choice(BRANDS)
    host += "." + rng.choice(GOOD_TLD)
    if rng.random() < .4: host = "www." + host
    path = "/".join(rng.choice(WORDS) for _ in range(rng.randint(0, 3)))
    url = ("https://" if rng.random() < .85 else "http://") + host + "/" + path
    if rng.random() < .15: url += "?id=" + str(rng.randint(1, 9999))
    if rng.random() < .05: url = f"https://{rng.choice(BRANDS)}.com/{rng.choice(KEYS)}"   # hard negatives
    return url

def phish(rng):
    kind = rng.random()
    brand = rng.choice(BRANDS)
    if kind < .25:                       # IP host
        host = ".".join(str(rng.randint(1, 255)) for _ in range(4))
    elif kind < .55:                     # brand in subdomain
        host = f"{brand}.{rng.choice(KEYS)}-{_rand(rng, rng.randint(3,6))}.{rng.choice(BAD_TLD)}"
    elif kind < .8:                      # hyphenated brand
        host = f"{brand}-{rng.choice(KEYS)}-{rng.choice(WORDS)}.{rng.choice(BAD_TLD+GOOD_TLD[:3])}"
    elif kind < .9:                      # shortener
        host = rng.choice(SHORT)
    else:                                # clean-looking (hard positive)
        host = f"{rng.choice(WORDS)}{rng.choice(WORDS)}.{rng.choice(GOOD_TLD)}"
    path = "/".join(rng.choice(KEYS + [_rand(rng, 8)]) for _ in range(rng.randint(1, 4)))
    url = ("http://" if rng.random() < .65 else "https://") + host + "/" + path
    if rng.random() < .3: url += f"?user={_rand(rng, 6)}&token={_rand(rng, 12)}"
    if rng.random() < .1: url = url.replace("://", "://" + brand + ".com@")
    return url

def generate(n=6000, seed=42, noise=0.03):
    rng = random.Random(seed)
    rows = [((phish(rng) if i % 2 else legit(rng)), i % 2) for i in range(n)]
    df = pd.DataFrame(rows, columns=["url", "label"]).sample(frac=1, random_state=seed)
    flip = df.sample(frac=noise, random_state=seed).index          # label noise
    df.loc[flip, "label"] = 1 - df.loc[flip, "label"]
    return df.reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    generate(a.n).to_csv("data/urls.csv", index=False); print("saved data/urls.csv")
