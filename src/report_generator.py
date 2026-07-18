import argparse
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import pandas as pd

import os

OUTPUT_DIR = "topics_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------
# Load required files
# ------------------------------------
def load_inputs(input_dir):
    clusters = pd.read_csv(os.path.join(input_dir, "clusters.csv"))
    summary = pd.read_csv(os.path.join(input_dir, "topic_summary.csv"))
    return clusters, summary


# ------------------------------------
# Add image with auto-scaling
# ------------------------------------
def add_image(story, path, width=6*inch):
    if os.path.exists(path):
        img = Image(path)
        img.drawWidth = width
        img.drawHeight = img.drawWidth * img.imageHeight / img.imageWidth
        story.append(img)
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph(f"[Missing image: {path}]", getSampleStyleSheet()["Normal"]))
        story.append(Spacer(1, 12))


# ------------------------------------
# MAIN REPORT CREATION
# ------------------------------------
def create_report(input_dir, output_file):
    clusters, summary = load_inputs(input_dir)

    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Topic Modeling Report</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    # Basic info
    story.append(Paragraph("<b>Dataset Information</b>", styles["Heading2"]))
    story.append(Paragraph(f"Total reviews analyzed: {len(clusters)}", styles["Normal"]))
    story.append(Paragraph(f"Total topics discovered: {summary['topic'].nunique()}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Cluster size distribution
    story.append(Paragraph("<b>Cluster Size Distribution</b>", styles["Heading2"]))
    add_image(story, os.path.join(input_dir, "cluster_size_distribution.png"))

    # UMAP Plot
    story.append(Paragraph("<b>UMAP Topic Visualization</b>", styles["Heading2"]))
    add_image(story, os.path.join(input_dir, "umap_clusters.png"))

    # Topic summaries
    story.append(Paragraph("<b>Topic Summaries</b>", styles["Heading2"]))

    for _, row in summary.iterrows():
        topic_id = row["topic"]
        topic_name = row.get("topic_name", f"Topic {topic_id}")
        top_words = row["top_words"]

        story.append(Paragraph(f"<b>Topic {topic_id} — {topic_name}</b>", styles["Heading3"]))
        story.append(Paragraph(f"Top Words: {top_words}", styles["Normal"]))

        top_img = os.path.join(input_dir, f"topic_{topic_id}_top_words.png")
        add_image(story, top_img)

        story.append(Spacer(1, 12))

    # Build PDF
    doc = SimpleDocTemplate(output_file, pagesize=A4)
    doc.build(story)

    print("\n🎉 PDF Report Generated!")
    print("Saved to:", output_file)


# ------------------------------------
# Entry point
# ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate PDF topic modeling report")
    parser.add_argument("--input_dir", default="topics_output")
    parser.add_argument("--output", default="topic_model_report.pdf")
    args = parser.parse_args()

    create_report(args.input_dir, args.output)


if __name__ == "__main__":
    main()

#python report_generator.py --input_dir topics_output --output job_review_report.pdf
