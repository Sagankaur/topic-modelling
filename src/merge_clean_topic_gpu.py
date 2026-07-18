"""
merge_clean_topic_gpu.py

GPU-optimized BERTopic pipeline for large job reviews.
Embeddings, UMAP, and HDBSCAN all run on GPU.
Optionally names topics using Ollama.
"""

import os
import argparse
import subprocess
import json
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from cuml.manifold import UMAP  # GPU UMAP
from cuml.cluster import HDBSCAN  # GPU HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ------------------ CONFIG ------------------
BASE_DIR = r"D:\DOMS job reviews"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "topic_model_gpu")
DEFAULT_INPUT = os.path.join(BASE_DIR, "job_reviews_clean.csv") #CHANGE this
OUTPUT_TOPIC_RESULTS = os.path.join(MODEL_SAVE_DIR, "topic_results.csv")   #D:\DOMS job reviews\topic_model should be here
OUTPUT_TOPIC_LABELS = os.path.join(MODEL_SAVE_DIR, "topic_labels.csv")

# GPU-optimized UMAP/HDBSCAN params
UMAP_PARAMS = dict(
    n_neighbors=40,
    n_components=5,      # lower dims for GPU
    min_dist=0.0,
    metric="cosine"
)

HDBSCAN_PARAMS = dict(
    min_cluster_size=60,  # increase for large dataset
    min_samples=10,
    cluster_selection_method="eom"
)

BERTOPIC_PARAMS = dict(
    n_gram_range=(1, 3),
    calculate_probabilities=True
)

OLLAMA_DEFAULT_MODEL = "llama2"

# ------------------ DEVICE ------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_embedder(model_name: str = "all-mpnet-base-v2"):
    device = get_device()
    print(f"Loading embedder '{model_name}' on device: {device}")
    return SentenceTransformer(model_name, device=device)

# ------------------ OLLAMA LABEL ------------------
def name_topic_with_ollama(words: List[str], ollama_model: str, max_words: int = 12) -> Tuple[str, str]:
    top_words = ", ".join(words[:max_words])
    prompt = f"""
You are an expert data analyst.

WORDS:
{top_words}

TASK:
1. Short academic topic name (4-6 words)
2. One-sentence description (12-25 words)
3. Output ONLY valid JSON: {{"label":"...","description":"..."}}
"""
    try:
        proc = subprocess.run(
            ["ollama", "run", ollama_model],
            input=prompt, capture_output=True, text=True, timeout=60
        )
        import re
        match = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return parsed.get("label",""), parsed.get("description","")
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        return lines[0][:80], lines[1][:240] if len(lines)>=2 else lines[0][:80], ""
    except Exception as e:
        print(f"⚠️ Ollama failed: {e}")
        return fallback_label(words)

def fallback_label(words: List[str]) -> Tuple[str,str]:
    label = " ".join(words[:3])
    desc = " / ".join(words[:6])
    return label, desc

# ------------------ RUN MODEL ------------------
def run_topic_model(input_csv: str, ollama_model: str = OLLAMA_DEFAULT_MODEL,
                    embedder_name: str = "all-mpnet-base-v2",
                    umap_params: dict = UMAP_PARAMS,
                    hdbscan_params: dict = HDBSCAN_PARAMS,
                    bertopic_params: dict = BERTOPIC_PARAMS,
                    save_dir: str = MODEL_SAVE_DIR):

    os.makedirs(save_dir, exist_ok=True)
    print("Loading data:", input_csv)
    df = pd.read_csv(input_csv, encoding="utf-8")

    if "text" not in df.columns:
        df["text"] = df.get("pros_clean","").fillna("") + " " + df.get("cons_clean","").fillna("")
    texts = df["text"].fillna("").astype(str).tolist()
    print(f"Total documents: {len(texts)}")

    # GPU embeddings
    embedder = load_embedder(embedder_name)
    emb_path = os.path.join(save_dir,"embeddings.npy")
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
        print("Loaded embeddings from disk.")
    else:
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=256)
        np.save(emb_path, embeddings)
        print(f"Saved embeddings → {emb_path}")

    # GPU UMAP & HDBSCAN
    umap_model = UMAP(**umap_params)
    clusterer = HDBSCAN(**hdbscan_params)

    # BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=clusterer,
        verbose=True,
        **bertopic_params
    )

    print("Fitting BERTopic model on GPU...")
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Save partial
    df["topic"] = topics
    if probs is not None:
        df["topic_prob"] = [p.max() if hasattr(p,"max") else (p if isinstance(p,(int,float)) else None) for p in probs]
    df.to_csv(os.path.join(save_dir,"topic_results_partial.csv"),index=False)
    topic_model.save(os.path.join(save_dir,"topic_model_partial"))

    # Metrics
    try: sil = silhouette_score(embeddings, topics)
    except: sil=None
    try: db = davies_bouldin_score(embeddings, topics)
    except: db=None

    print("Silhouette:", sil)
    print("Davies-Bouldin:", db)

    # Topic labels via Ollama
    labels_records=[]
    for tid, items in topic_model.get_topics().items():
        if tid==-1: continue
        words = [w for w,_ in items]
        label, desc = name_topic_with_ollama(words, ollama_model)
        labels_records.append({
            "topic_id": tid,
            "label": label,
            "description": desc,
            "top_words": ", ".join(words[:15]),
            "count": int(topic_model.get_topic_info().loc[topic_model.get_topic_info()["Topic"]==tid,"Count"].values[0])
                     if not topic_model.get_topic_info().loc[topic_model.get_topic_info()["Topic"]==tid,"Count"].empty
                     else None
        })
        pd.DataFrame(labels_records).to_csv(os.path.join(save_dir,"topic_labels_partial.csv"),index=False)

    # Save final results
    df.to_csv(OUTPUT_TOPIC_RESULTS,index=False)
    pd.DataFrame(labels_records).to_csv(OUTPUT_TOPIC_LABELS,index=False)
    topic_model.save(os.path.join(save_dir,"bertopic_model"))
    print(f"Saved final results → {OUTPUT_TOPIC_RESULTS}, {OUTPUT_TOPIC_LABELS}")

    return {
        "model_dir": save_dir,
        "silhouette": sil,
        "davies_bouldin": db,
        "n_topics": len(labels_records)
    }

# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GPU BERTopic + Ollama")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--ollama_model", type=str, default=OLLAMA_DEFAULT_MODEL)
    parser.add_argument("--embedder", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--min_topic_size", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    HDBSCAN_PARAMS["min_cluster_size"] = args.min_topic_size
    metrics = run_topic_model(
        input_csv=os.path.join(BASE_DIR,args.input) if not os.path.isabs(args.input) else args.input,
        ollama_model=args.ollama_model,
        embedder_name=args.embedder,
        umap_params=UMAP_PARAMS,
        hdbscan_params=HDBSCAN_PARAMS,
        bertopic_params=BERTOPIC_PARAMS,
        save_dir=args.save_dir
    )
    print("Done. Summary:")
    print(json.dumps(metrics, indent=2))
