import csv, json, collections
import pandas as pd
from sklearn.model_selection import train_test_split
from common import MAX_LEN

UCI_PATH  = "data/SMSSpamCollection"
MEND_PATH = "data/Mendeley_dataset.csv"
SENT_PATH = "data/sentinel_preprocessed_clean.csv"

UCI_SPAM_AS  = "scam"   # "scam" or "drop"
MEND_SPAM_AS = "scam"   # "scam" or "drop"
SEED = 42

SENT_TYPES = {0: "legit", 1: "government_impersonation", 2: "financial_fraud",
              3: "bank_phishing", 4: "romance_parcel_sextortion"}


def read_any(path, **kw):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue


def find_col(df, *names):
    low = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def load_uci():
    df = read_any(UCI_PATH, sep="\t", header=None, names=["raw", "text"],
                  quoting=csv.QUOTE_NONE)
    raw = df["raw"].astype(str).str.strip().str.lower()
    out = pd.DataFrame({"text": df["text"], "source": "uci"})
    out["label"] = raw.map({"ham": 0, "spam": 1 if UCI_SPAM_AS == "scam" else None})
    out["scam_type"] = out["label"].map({0: "legit", 1: "unlabeled"})
    return out


def load_mendeley():
    df = read_any(MEND_PATH)
    lab = df[find_col(df, "label", "labels", "class")].astype(str).str.strip().str.lower()
    print("mendeley raw labels:", lab.value_counts().to_dict())
    out = pd.DataFrame({"text": df[find_col(df, "text", "message", "sms")], "source": "mendeley"})
    spam = 1 if MEND_SPAM_AS == "scam" else None
    out["label"] = lab.map({"ham": 0, "spam": spam, "smishing": 1})
    out["scam_type"] = out["label"].map({0: "legit", 1: "unlabeled"})
    out.loc[lab == "smishing", "scam_type"] = "smishing"
    return out


def load_sentinel():
    df = read_any(SENT_PATH)
    print("sentinel columns:", list(df.columns))
    y = pd.to_numeric(df[find_col(df, "label", "labels", "category")], errors="coerce")
    out = pd.DataFrame({"text": df[find_col(df, "text", "message")], "source": "sentinel"})
    out["scam_type"] = y.map(SENT_TYPES)
    out["label"] = (y > 0).astype("float").where(y.notna())
    return out


parts = [load_uci(), load_mendeley(), load_sentinel()]
df = pd.concat(parts, ignore_index=True)
print("\nloaded rows per source:\n", df["source"].value_counts().to_string())

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df["label"] = df["label"].astype(int)
print("after dropping unmapped/empty:", len(df))

# dedupe across ALL sources on a normalized key; drop conflicting labels
df["key"] = (df["text"].str.lower()
             .str.replace(r"\d", "0", regex=True)
             .str.replace(r"\s+", " ", regex=True).str.strip())
nconf = df.groupby("key")["label"].nunique()
bad = nconf[nconf > 1].index
print("conflicting-label keys dropped:", len(bad))
df = df[~df["key"].isin(bad)]
before = len(df)
df = df.drop_duplicates("key").drop(columns="key")
print("duplicates removed:", before - len(df), "| final rows:", len(df))
print(pd.crosstab(df["source"], df["label"]).to_string())

# split, stratified on source+label
strat = df["source"] + "_" + df["label"].astype(str)
train, temp = train_test_split(df, test_size=0.2, stratify=strat, random_state=SEED)
val, test = train_test_split(temp, test_size=0.5,
                             stratify=(temp["source"] + "_" + temp["label"].astype(str)),
                             random_state=SEED)
cols = ["text", "label", "source", "scam_type"]
for name, d in [("train", train), ("val", val), ("test", test)]:
    d[cols].to_csv(f"data/{name}.csv", index=False)
    print(f"\n{name}: {len(d)}")
    print(pd.crosstab(d["source"], d["label"]).to_string())

# vocab from TRAIN only; 0=PAD, 1=UNK
counts = collections.Counter(c for t in train["text"] for c in t.lower())
chars = [c for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]
vocab = {c: i + 2 for i, c in enumerate(chars)}
with open("models/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=0)
print("\nvocab size (incl PAD, UNK):", len(vocab) + 2)

print("\nlength percentiles per source (train):")
print(train.groupby("source")["text"].apply(lambda s: s.str.len().quantile([.5, .95, .99]).round().tolist()).to_string())
print("train longer than %d chars: %.1f%%" % (MAX_LEN, (train["text"].str.len() > MAX_LEN).mean() * 100))