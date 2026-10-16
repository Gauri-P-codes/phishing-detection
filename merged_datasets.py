import sys
import pandas as pd
from src.data_gen import generate

real_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
syn_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

real = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"]).rename(columns={"URL": "url"})
real["label"] = 1 - real["label"]                     # 1 = phishing, as in the project
real = real.drop_duplicates("url")
real = pd.concat([g.sample(min(len(g), real_n // 2), random_state=42) for _, g in real.groupby("label")])

syn = generate(syn_n)                                 # already 1 = phishing

df = pd.concat([real, syn]).drop_duplicates("url").sample(frac=1, random_state=42)
df.to_csv("data/urls.csv", index=False)
print(f"Saved {len(df)} rows | phishing ratio = {df['label'].mean():.2f}")