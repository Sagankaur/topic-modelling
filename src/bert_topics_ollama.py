import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan
from openai import OpenAI
from openpyxl import load_workbook

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_CSV = "cleaned_glassdoor_reviews.csv"
OUTPUT_XLSX = "glassdoor_ProsCons_Topics.xlsx"

EMB_CACHE_DIR = "embedding_cache"
os.makedirs(EMB_CACHE_DIR, exist_ok=True)

# -------------------------------------------------------------------
# OPENAI
# -------------------------------------------------------------------
# client = OpenAI()


def embed_texts(texts, cache_path):
    """Embed text with caching."""
    if os.path.exists(cache_path):
        print(f"Loading embedding cache: {cache_path}")
        data = np.load(cache_path)
        return data["embeddings"]

    print("Generating embeddings...")
    BATCH = 128
    all_embeds = []

    for i in tqdm(range(0, len(texts), BATCH)):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        vecs = [x.embedding for x in resp.data]
        all_embeds.extend(vecs)

    arr = np.array(all_embeds, dtype=float)
    np.savez(cache_path, embeddings=arr)
    return arr

# -------------------------------------------------------------------
# CLUSTERING + TOPIC EXTRACTION
# -------------------------------------------------------------------
def reduce_umap(vectors, save_path):
    """UMAP with partial save + resume."""
    if os.path.exists(save_path):
        print("Loading saved UMAP embedding:", save_path)
        return np.load(save_path)

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, n_components=5, random_state=42)
    umap_result = reducer.fit_transform(vectors)

    np.save(save_path, umap_result)
    return umap_result


def run_hdbscan(vectors, save_path):
    """HDBSCAN with save + resume."""
    if os.path.exists(save_path):
        print("Loading saved HDBSCAN labels:", save_path)
        return np.load(save_path)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_samples=20, 
        min_cluster_size=50,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(vectors)

    np.save(save_path, labels)
    return labels


def extract_topic_keywords(texts, labels, save_path):
    """Extract representative topic words for each cluster."""
    if os.path.exists(save_path):
        print("Loading saved topic keywords:", save_path)
        with open(save_path, "r") as f:
            return json.load(f)

    df = pd.DataFrame({"text": texts, "label": labels})
    clusters = df["label"].unique()

    topic_keywords = {}

    for c in clusters:
        if c == -1:
            continue
        group = df[df["label"] == c]["text"]

        tfidf = TfidfVectorizer(stop_words="english", max_features=20)
        X = tfidf.fit_transform(group)
        words = tfidf.get_feature_names_out()

        topic_keywords[int(c)] = list(words[:10])

    # Save JSON
    with open(save_path, "w") as f:
        json.dump(topic_keywords, f, indent=2)

    return topic_keywords


# -------------------------------------------------------------------
# EXCEL WRITER (Incremental)
# -------------------------------------------------------------------
def write_excel_sheet(df, sheet_name):
    """Append or replace a sheet inside XLSX safely."""
    if not os.path.exists(OUTPUT_XLSX):
        # create new file
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        return

    # append/replace sheet
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def process_block(texts, prefix):
    """
    Handles Pros or Cons:
      embeddings → UMAP → HDBSCAN → keywords → write sheets
    """
    print(f"\n=== STARTING {prefix.upper()} PROCESSING ===")

    # ------------------------------
    # EMBEDDINGS
    # ------------------------------
    emb_path = f"{EMB_CACHE_DIR}/{prefix}_embeddings.npz"
    embeddings = embed_texts(texts, emb_path)

    # ------------------------------
    # UMAP
    # ------------------------------
    umap_path = f"{EMB_CACHE_DIR}/{prefix}_umap.npy"
    umap_vecs = reduce_umap(embeddings, umap_path)

    # ------------------------------
    # HDBSCAN
    # ------------------------------
    labels_path = f"{EMB_CACHE_DIR}/{prefix}_labels.npy"
    labels = run_hdbscan(umap_vecs, labels_path)

    # ------------------------------
    # TOPIC KEYWORDS
    # ------------------------------
    keyword_json = f"{prefix}_cluster_keywords.json"
    keywords = extract_topic_keywords(texts, labels, keyword_json)

    # ----------------------------------------------------------------
    # Build topic name mapping
    # ----------------------------------------------------------------
    topic_names = {
        int(k): "_".join(v[:3]) if v else "outlier_noise"
        for k, v in keywords.items()
    }
    topic_names[-1] = "outlier_noise"

    # ----------------------------------------------------------------
    # WRITE SHEET: SUMMARY
    # ----------------------------------------------------------------
    cluster_counts = pd.DataFrame({
        "Topic": list(topic_names.keys()),
        "Topic_Name": [topic_names[k] for k in topic_names],
        "Count": [int(sum(labels == k)) for k in topic_names]
    })
    write_excel_sheet(cluster_counts, f"{prefix}_Topics_Summary")

    # ----------------------------------------------------------------
    # WRITE SHEET: WITH CLUSTERS
    # ----------------------------------------------------------------
    df_clusters = pd.DataFrame({
        prefix.lower() + "_text": texts,
        "cluster": labels,
        "topic_name": [topic_names.get(int(x), "outlier_noise") for x in labels]
    })
    write_excel_sheet(df_clusters, f"{prefix}_With_Cluster")

    # ----------------------------------------------------------------
    # WRITE SHEET: KEYWORDS
    # ----------------------------------------------------------------
    key_rows = []
    for c, words in keywords.items():
        key_rows.append({
            "Topic": c,
            "Topic_Name": topic_names[c],
            "Keywords": ", ".join(words)
        })
    df_keywords = pd.DataFrame(key_rows)
    write_excel_sheet(df_keywords, f"{prefix}_Topic_Keywords")

    print(f"✓ Completed {prefix.upper()}")
    return labels, keywords


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV)

    pros = df["pros_clean"].fillna("").tolist()
    cons = df["cons_clean"].fillna("").tolist()

    # process Pros
    pros_labels, pros_keywords = process_block(pros, "Pros")

    # process Cons
    cons_labels, cons_keywords = process_block(cons, "Cons")

    print("\n=== ALL DONE ===")
    print("Results saved to:", OUTPUT_XLSX)
