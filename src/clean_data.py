import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORD_FILE = r"D:\DOMS job reviews\top_phrases.xlsx"


# ---------------------------------------------------------
# 1. Load stopwords from Excel (two sheets!)
# ---------------------------------------------------------
def load_stopwords():

    # Load the two sheets
    pros_df = pd.read_excel(STOPWORD_FILE, sheet_name="Top Pros Phrases")
    cons_df = pd.read_excel(STOPWORD_FILE, sheet_name="Top Cons Phrases")

    # Normalize text
    pros_df["Pros Phrase"] = pros_df["Pros Phrase"].astype(str).str.lower().str.strip()
    cons_df["Cons Phrase"] = cons_df["Cons Phrase"].astype(str).str.lower().str.strip()
    pros_df["Label"] = pros_df["Label"].astype(str).str.strip()
    cons_df["Label"] = cons_df["Label"].astype(str).str.strip()

    # Labels to use as stopwords
    PROS_REMOVE_LABELS = ["HR", "Management", "Generic Praise"]
    CONS_REMOVE_LABELS = ["HR", "Management", "Generic Complaint"]

    # Extract stopwords from relevant labels
    pros_sw = set(
        pros_df[pros_df["Label"].isin(PROS_REMOVE_LABELS)]["Pros Phrase"].tolist()
    )

    cons_sw = set(
        cons_df[cons_df["Label"].isin(CONS_REMOVE_LABELS)]["Cons Phrase"].tolist()
    )

    print(f"Loaded {len(pros_sw)} PROS stopwords")
    print(f"Loaded {len(cons_sw)} CONS stopwords")

    return pros_sw, cons_sw


# Generic bad filler phrases (OPTIONAL)
GENERIC_BAD_PHRASES = [
    r"\bnot applicable\b",
    r"\blike this\b",
    r"\bstuff like that\b",
    r"\bethc\b",
    r"\bna\b",
    r"\bnone\b",
    r"\bnothing i can think of\b",
    r"\bnone that i can think of\b",
    r"\bna na\b",
    r"\bno cons\b",
]

lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------
# 2. Clean text
# ---------------------------------------------------------
def clean_text(text: str, stopwords: set) -> str:
    if not isinstance(text, str):
        return ""

    t = text.lower().strip()

    # Remove emojis & non-ASCII
    t = re.sub(r"[^\x00-\x7F]+", " ", t)

    # Remove URLs
    t = re.sub(r"http\S+|www\.\S+", " ", t)

    # Remove generic filler phrases
    for p in GENERIC_BAD_PHRASES:
        t = re.sub(p, " ", t)

    # Remove punctuation
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)

    # Remove standalone numbers
    t = re.sub(r"\b\d+\b", " ", t)

    # Collapse spaces
    t = re.sub(r"\s+", " ", t).strip()

    # Tokenize
    tokens = t.split()

    # Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords]

    # Remove tiny tokens
    tokens = [w for w in tokens if len(w) > 2]

    return " ".join(tokens)


# ---------------------------------------------------------
# 3. Apply cleaning to DataFrame
# ---------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    pros_sw, cons_sw = load_stopwords()

    print("Cleaning PROS...")
    df["pros_clean"] = df["pros"].fillna("").apply(lambda x: clean_text(x, pros_sw))

    print("Cleaning CONS...")
    df["cons_clean"] = df["cons"].fillna("").apply(lambda x: clean_text(x, cons_sw))

    # Remove empty combined rows
    before = len(df)
    df = df[(df["pros_clean"] != "") | (df["cons_clean"] != "")]
    after = len(df)

    print(f"Removed {before - after} empty rows")

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# 4. Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = r"D:\DOMS job reviews\filtered_2021_sector_IT___Technology___Software.csv" #CHaNGE THIS
    OUTPUT_FILE = r"D:\DOMS job reviews\job_reviews_clean.csv"

    print("📘 Loading raw dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("🧹 Cleaning dataset (pros + cons)...")
    df = clean_dataframe(df)

    print("💾 Saving cleaned dataset to:", OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Cleaning complete!")
