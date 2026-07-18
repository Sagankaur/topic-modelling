import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bertopic import BERTopic
from umap import UMAP

BASE_DIR = r"D:\DOMS job reviews\topic_model_llama_energy_pros"

MODEL_PATH = os.path.join(BASE_DIR, "bertopic_model")
LABELS_PATH = os.path.join(BASE_DIR, "topic_labels.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
CLUSTERS_PATH = os.path.join(BASE_DIR, "topics_output", "clusters.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "topics_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# LOAD MODEL + LABELS
# ---------------------------------------------------------
print("Loading BERTopic model…")
topic_model = BERTopic.load(MODEL_PATH)

print("Loading labels…")
labels_df = pd.read_csv(LABELS_PATH)

print("Loading embeddings…")
embeddings = np.load(EMBEDDINGS_PATH)

# ─────────────────────────────────────────────────────────
# CHECK IF clusters.csv exists → if not, recompute UMAP
# ─────────────────────────────────────────────────────────
if os.path.exists(CLUSTERS_PATH):
    print("✔ clusters.csv found, loading it…")
    clusters_df = pd.read_csv(CLUSTERS_PATH)

else:
    print("❗ clusters.csv missing — recomputing UMAP for visualization…")

    # Extract topics from the saved model
    topics, _ = topic_model.transform([None] * embeddings.shape[0])

    print("Running UMAP on embeddings…")
    umap_model = UMAP(
        n_neighbors=50,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    umap_emb = umap_model.fit_transform(embeddings)

    print("Building new clusters dataframe…")
    clusters_df = pd.DataFrame({
        "topic": topics,
        "umap_x": umap_emb[:, 0],
        "umap_y": umap_emb[:, 1],
    })

    clusters_df.to_csv(CLUSTERS_PATH, index=False)
    print(f"✔ Saved recomputed clusters.csv → {CLUSTERS_PATH}")

# ---------------------------------------------------------
# MERGE LABELS WITH CLUSTERS
# ---------------------------------------------------------
clusters_df = clusters_df.merge(labels_df, left_on="topic", right_on="topic_id", how="left")

# ---------------------------------------------------------
# 1. UMAP SCATTER PLOT
# ---------------------------------------------------------
print("Creating UMAP scatter plot…")
plt.figure(figsize=(11, 8))

scatter = plt.scatter(
    clusters_df["umap_x"],
    clusters_df["umap_y"],
    c=clusters_df["topic"],
    cmap="tab20",
    s=10,
    alpha=0.7,
)

plt.colorbar(scatter, label="Topic ID")
plt.title("UMAP Visualization of Reviews by Topic")
plt.xlabel("UMAP X")
plt.ylabel("UMAP Y")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_topics.png"), dpi=300)
plt.close()

print(f"✔ Saved → {os.path.join(OUTPUT_DIR, 'umap_topics.png')}")


# ---------------------------------------------------------
# 2. TOPIC SIZE BAR CHART
# ---------------------------------------------------------
print("Creating topic frequency chart…")
topic_sizes = clusters_df["topic"].value_counts().sort_index()

plt.figure(figsize=(12, 5))
plt.bar(topic_sizes.index.astype(str), topic_sizes.values)
plt.xticks(rotation=90)
plt.title("Topic Frequency (Number of Reviews per Topic)")
plt.xlabel("Topic ID")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "topic_sizes.png"), dpi=300)
plt.close()

print(f"✔ Saved → {os.path.join(OUTPUT_DIR, 'topic_sizes.png')}")


# ---------------------------------------------------------
# 3. TOPIC SUMMARY TABLE
# ---------------------------------------------------------
print("Creating topic summary table…")
summary_records = []

for _, row in labels_df.iterrows():
    tid = row["topic_id"]
    words = row["top_words"]
    label = row["label"]
    desc = row["description"]

    count = clusters_df[clusters_df["topic"] == tid].shape[0]

    summary_records.append({
        "topic_id": tid,
        "label": label,
        "description": desc,
        "count": count,
        "top_words": words,
    })

summary_df = pd.DataFrame(summary_records)
summary_file = os.path.join(OUTPUT_DIR, "topic_summary_full.csv")
summary_df.to_csv(summary_file, index=False)

print(f"✔ Saved → {summary_file}")


# ---------------------------------------------------------
# 4. INTERACTIVE HTML VISUALIZATIONS
# ---------------------------------------------------------
print("Generating interactive HTML visualizations…")

try:
    topic_model.visualize_barchart().write_html(os.path.join(OUTPUT_DIR, "interactive_barchart.html"))
    topic_model.visualize_hierarchy().write_html(os.path.join(OUTPUT_DIR, "interactive_hierarchy.html"))
    topic_model.visualize_topics().write_html(os.path.join(OUTPUT_DIR, "interactive_topics.html"))

    print("✔ Saved interactive visualizations:")
    print("  - interactive_barchart.html")
    print("  - interactive_hierarchy.html")
    print("  - interactive_topics.html")

except Exception as e:
    print("⚠ Could not create interactive HTML: ", e)

print("\n🎉 ALL VISUALIZATIONS READY!\n")
