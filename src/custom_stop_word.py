import pandas as pd
import subprocess
import argparse
import time
import os

# -----------------------------------------------------
# RUN OLLAMA LOCALLY (NO API)
# -----------------------------------------------------
def ask_ollama_local(phrase, model="llama3.1"):

    LABELS = [
        "HR",
        "Management",
        "Coworkers",
        "Work Environment",
        "Work-Life Balance",
        "Pay & Benefits",
        "Career Growth",
        "Product Quality",
        "Product Features",
        "Brand Reputation",
        "Generic Praise",
        "Generic Complaint",
        "Other"
    ]

    prompt = f"""
Classify the following phrase into EXACTLY one label:

{LABELS}

Return ONLY the label.

Phrase: "{phrase}"
Label:
"""

    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(prompt, timeout=60)

        if stderr:
            print("Ollama stderr:", stderr.strip())

        output = stdout.strip()

        # clean
        output_lower = output.lower()

        # match one of the correct labels
        for label in LABELS:
            if label.lower() in output_lower:
                return label

        return "Other"

    except Exception as e:
        print(f"Error running Ollama: {e}")
        return "Other"


# -----------------------------------------------------
# CLASSIFY ALL PHRASES (WRITE INTO SAME FILE)
# -----------------------------------------------------
def classify_phrases(excel_path, model):

    print(f"\n📘 Loading Excel file: {excel_path}")
    sheets = pd.read_excel(excel_path, sheet_name=None)

    STOPWORD_LABELS = {
        "HR",
        "Management",
        "Coworkers",
        "Work Environment",
        "Work-Life Balance",
        "Pay & Benefits",
        "Career Growth",
        "Generic Praise",
        "Generic Complaint"
    }

    save_every = 20   # partial save

    writer = pd.ExcelWriter(excel_path, engine="openpyxl", mode='w')

    # -------------------------
    # Process each sheet
    # -------------------------
    for sheet_name, df in sheets.items():
        print(f"\n📄 Processing sheet: {sheet_name}")

        # Ensure columns exist
        if "Label" not in df.columns:
            df["Label"] = ""

        if "IsStopword" not in df.columns:
            df["IsStopword"] = ""

        # Resume logic
        unclassified = df[df["Label"].isna() | (df["Label"] == "")]
        if len(unclassified) == 0:
            print(f"✔ Sheet '{sheet_name}' already completed.")
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            continue

        start_index = unclassified.index[0]
        print(f"🔎 Resuming from row {start_index + 1}")

        # Classification
        for i in range(start_index, len(df)):
            phrase = str(df.at[i, df.columns[0]])

            label = ask_ollama_local(phrase, model=model)
            df.at[i, "Label"] = label
            df.at[i, "IsStopword"] = "Yes" if label in STOPWORD_LABELS else "No"

            print(f"{i+1}/{len(df)} → {phrase} → {label}")

            # partial save every X rows
            if i % save_every == 0:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                writer.book.save(excel_path)
                print("💾 Partial progress saved.")

            time.sleep(0.05)

        df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()
    print("\n✅ DONE — All sheets updated inside the same Excel file!")


# -----------------------------------------------------
# CLI
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="llama3.1")
    args = parser.parse_args()

    classify_phrases(args.input, args.model)

#python custom_stop_word.py --input "top_phrases.xlsx" --model "llama3.1"
