"""
merge_clean_topic_llama.py

LLaMA-based topic extraction pipeline (no BERTopic).
- Computes embeddings with SentenceTransformers (fast).
- Dimensionality reduction with UMAP.
- Clustering with HDBSCAN (preferred) or KMeans fallback.
- Uses Ollama (local LLaMA) to name each cluster and produce a short description.
- Saves partial results and visualisation data.

Usage:
    python merge_clean_topic_llama.py --input job_reviews_clean_energy_2021.csv --ollama_model llama2 --n_clusters 40

Notes:
- Make sure `ollama` CLI is available and your model supports generation.
- This is designed to be resilient: if HDBSCAN isn't installed or fails, it falls back to KMeans.
"""

import os
import argparse
import subprocess
import json
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Try to import hdbscan; if unavailable, we'll fallback to KMeans
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# ---------------------- DEFAULT CONFIG ----------------------
BASE_DIR = r"D:\DOMS job reviews"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "topic_model_llama")

DEFAULT_INPUT = os.path.join(BASE_DIR, "job_reviews_clean.csv")
OUTPUT_TOPIC_RESULTS = os.path.join(MODEL_SAVE_DIR, "topic_results.csv")
OUTPUT_TOPIC_LABELS = os.path.join(MODEL_SAVE_DIR, "topic_labels.csv")

UMAP_PARAMS = dict(
    n_neighbors=50,
    n_components=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

HDBSCAN_PARAMS = dict(
    min_cluster_size=20,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# -------------------------------------------------------
# UTIL
# -------------------------------------------------------

def get_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_embedder(model_name: str = "all-mpnet-base-v2"):
    device = get_device()
    print(f"Loading embedder '{model_name}' on device: {device}")
    return SentenceTransformer(model_name, device=device)

# -------------------------------------------------------
# CALL OLLAMA TO NAME A TOPIC
# -------------------------------------------------------

def name_topic_with_ollama(cluster_id: int, sample_texts: List[str], ollama_model: str, max_words: int = 10) -> Tuple[int, str, str]:
    """
    Ask Ollama to provide a label and one-sentence description for a cluster.
    Returns: (cluster_id, label, description)
    """
    sample_preview = "\n\n".join(sample_texts[:5])
    prompt = f"""
You are an expert data analyst.

Below are several representative review excerpts belonging to a single latent topic (cluster).

EXCERPTS:
{sample_preview}

TASK:
1) Propose a short, academic-sounding topic label (4-6 words).
2) Write a concise one-sentence description (12-25 words) explaining the theme.
3) Output ONLY valid JSON with exactly these keys: label, description.

Respond with STRICTLY JSON only. No extra commentary.
"""
    try:
        proc = subprocess.run(
            ["ollama", "run", ollama_model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60
        )
        out = proc.stdout.strip()
        # Save raw output for debugging
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        with open(os.path.join(MODEL_SAVE_DIR, f"ollama_cluster_{cluster_id}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(out)

        import re
        match = re.search(r"\{.*\}", out, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return cluster_id, parsed.get("label", ""), parsed.get("description", "")

        # fallback simple parsing
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if len(lines) >= 2:
            return cluster_id, lines[0][:80], lines[1][:240]
        elif len(lines) == 1:
            return cluster_id, lines[0][:80], ""
        raise ValueError("No usable output from Ollama")
    except Exception as e:
        print(f"⚠️ Ollama failed for cluster {cluster_id}: {e}")
        # fallback: derive label from top words of sample texts
        words = _top_words_from_texts(sample_texts, topn=6)
        label = " ".join(words[:4])
        desc = " / ".join(words[:8])
        return cluster_id, label, desc

# small helper to extract frequent words
import re
from collections import Counter

def _top_words_from_texts(texts: List[str], topn: int = 8) -> List[str]:
    text = " ".join(texts).lower()
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    stop = set(["the","and","for","with","from","that","this","have","they","were","their","but","are","was"]) 
    tokens = [t for t in tokens if t not in stop]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(topn)]

# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def run_topic_model_llama(input_csv: str,
                          ollama_model: str = "llama2",
                          embedder_name: str = "all-mpnet-base-v2",
                          umap_params: dict = UMAP_PARAMS,
                          hdbscan_params: dict = HDBSCAN_PARAMS,
                          save_dir: str = MODEL_SAVE_DIR,
                          n_clusters: Optional[int] = 40):

    os.makedirs(save_dir, exist_ok=True)
    print("Loading cleaned data:", input_csv)
    df = pd.read_csv(input_csv, encoding="utf-8")

    if "text" not in df.columns:
        # df["text"] = df.get("pros_clean", "").fillna("") + " " + df.get("cons_clean", "").fillna("")
        df["text"] = df.get("cons_clean", "").fillna("")

    texts = df["text"].fillna("").astype(str).tolist()
    print(f"Total documents: {len(texts)}")

    embedder = load_embedder(embedder_name)

    emb_path = os.path.join(save_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        print(f"📌 Found existing embeddings → {emb_path}")
        embeddings = np.load(emb_path)
    else:
        print("🔍 Computing embeddings...")
        batch_size = 128 if get_device() == "cuda" else 32
        embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        np.save(emb_path, embeddings)
        print(f"💾 Saved embeddings → {emb_path}")

    # normalize embeddings for clustering stability
    embeddings = normalize(embeddings)

    # dimensionality reduction for visualization (and optionally clustering)
    umap_model = UMAP(**umap_params)
    umap_embeddings = umap_model.fit_transform(embeddings)

    # CLUSTERING: prefer HDBSCAN if available, else KMeans
    cluster_labels = None
    if HDBSCAN_AVAILABLE:
        try:
            print("Clustering with HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            clusterer.fit(umap_embeddings)
            cluster_labels = clusterer.labels_
            # HDBSCAN can label noise as -1
        except Exception as e:
            print("HDBSCAN failed, falling back to KMeans:", e)
            HDBSCAN_fallback = True
        else:
            HDBSCAN_fallback = False
    else:
        HDBSCAN_fallback = True

    if HDBSCAN_fallback:
        # KMeans fallback
        nk = n_clusters if n_clusters and n_clusters > 0 else 40
        print(f"Clustering with KMeans (n_clusters={nk})...")
        km = KMeans(n_clusters=nk, random_state=42)
        cluster_labels = km.fit_predict(umap_embeddings)

    df["topic"] = cluster_labels

    # compute simple metrics on a sample
    sample_idx = np.random.choice(len(embeddings), size=min(5000, len(embeddings)), replace=False)
    try:
        sil = silhouette_score(embeddings[sample_idx], cluster_labels[sample_idx])
    except Exception:
        sil = None
    try:
        db = davies_bouldin_score(embeddings[sample_idx], cluster_labels[sample_idx])
    except Exception:
        db = None

    print("Silhouette:", sil, "Davies-Bouldin:", db)

    # Prepare cluster -> representative texts mapping
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(int(label), []).append(idx)

    # Build label records by asking Ollama to name each cluster
    labels_records = []

    def process_cluster(cluster_id):
        if cluster_id == -1:
            # optional: skip noise cluster naming
            return None
        idxs = clusters.get(cluster_id, [])
        # sample up to 12 representative documents
        sample_texts = [texts[i] for i in idxs[:12]]
        cid, label, desc = name_topic_with_ollama(cluster_id, sample_texts, ollama_model)
        return {
            "topic_id": int(cluster_id),
            "label": label,
            "description": desc,
            "count": len(idxs),
            "sample_idx": idxs[:12]
        }

    cluster_ids = sorted(list(clusters.keys()))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for res in executor.map(process_cluster, cluster_ids):
            if res:
                labels_records.append(res)
                # save partial labels
                pd.DataFrame(labels_records).to_csv(os.path.join(save_dir, "topic_labels_partial.csv"), index=False)

    # attach label to dataframe via merge
    labels_df = pd.DataFrame(labels_records)
    if not labels_df.empty:
        # rename topic id column to match df["topic"] values
        labels_df_ren = labels_df.rename(columns={"topic_id": "topic"})
        df = df.merge(labels_df_ren[["topic", "label", "description", "count"]], on="topic", how="left")

    # save results
    df.to_csv(OUTPUT_TOPIC_RESULTS, index=False)
    labels_df.to_csv(OUTPUT_TOPIC_LABELS, index=False)
    print(f"Saved final topic results → {OUTPUT_TOPIC_RESULTS}")
    print(f"Saved final topic labels → {OUTPUT_TOPIC_LABELS}")

    # save visualization artifacts
    vis_dir = os.path.join(save_dir, "topics_output")
    os.makedirs(vis_dir, exist_ok=True)
    try:
        clusters_df = pd.DataFrame({
            "review_text": texts,
            "topic": cluster_labels,
            "umap_x": umap_embeddings[:, 0],
            "umap_y": umap_embeddings[:, 1]
        })
        clusters_df.to_csv(os.path.join(vis_dir, "clusters.csv"), index=False)
    except Exception as e:
        print("Failed to create clusters.csv:", e)

    topic_summary_list = []
    for rec in labels_records:
        topic_summary_list.append({"topic": rec["topic_id"], "label": rec["label"], "description": rec["description"], "count": rec["count"]})
    pd.DataFrame(topic_summary_list).sort_values("topic").to_csv(os.path.join(vis_dir, "topic_summary.csv"), index=False)

    return {
        "model_dir": save_dir,
        "silhouette": sil,
        "davies_bouldin": db,
        "n_topics": len(labels_records),
    }

# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA-based topic extraction for job reviews.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--ollama_model", type=str, default="llama2")
    parser.add_argument("--embedder", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR)
    parser.add_argument("--n_clusters", type=int, default=40)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL_SAVE_DIR = args.save_dir

    metrics = run_topic_model_llama(
        input_csv=os.path.join(BASE_DIR, args.input) if not os.path.isabs(args.input) else args.input,
        ollama_model=args.ollama_model,
        embedder_name=args.embedder,
        umap_params=UMAP_PARAMS,
        hdbscan_params=HDBSCAN_PARAMS,
        save_dir=MODEL_SAVE_DIR,
        n_clusters=args.n_clusters
    )

    print("Done. Summary:")
    print(json.dumps(metrics, indent=2, default=str))
