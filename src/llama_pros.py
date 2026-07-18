#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from tqdm import tqdm
import subprocess
import os

# --- Configuration ---
# NOTE: Ensure the file paths below are correct for your system.
# BASE_DIR = r"D:\IIT\bert\fedx"
# --- Configuration ---
# Correct Windows path
BASE_DIR = r"D:\DOMS job reviews"

# EXCEL_FILE_NAME = "updated_glassdoor_data.csv"
EXCEL_FILE_NAME = "filtered_sector.csv" #from D:\DOMS job reviews\merge_with_sector.py
STOPWORD_FILE = r"D:\DOMS job reviews\classified_stopwords.csv" # D:\DOMS job reviews\custom_stop_word.py

OUTPUT_FILE_NAME = "glassdoor_ProsCons_Prediction.xlsx"

EXCEL_FILE = os.path.join(BASE_DIR, EXCEL_FILE_NAME)
OUTPUT_FILE = os.path.join(BASE_DIR, OUTPUT_FILE_NAME)
# ---------------------

# ---------------------

def run_llama_local(prompt: str, timeout: int = 90) -> str:
    """
    Runs the Llama model locally via the ollama CLI.
    """
    try:
        # NOTE: Make sure 'llama3.1:8b' is installed in your ollama setup
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        output = result.stdout.strip()
        print(f"\n/Prediction:{output}")
        return output
    
    except subprocess.TimeoutExpired:
        print("LLaMA timeout")
        return ""
    except Exception as e:
        print(f"LLaMA error: {e}")
        return ""

def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filters the dataset to include only rows where the 'date' column matches the given year.
    """
    if "date" not in df.columns:
        raise ValueError("The 'date' column is missing from the dataset.")

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by the given year
    filtered_df = df[df["date"].dt.year == year].reset_index(drop=True)

    print(f"Rows after filtering for YEAR {year}: {len(filtered_df)}")

    return filtered_df

def remove_stopword_rows(df, stopword_file):
    """
    Remove rows where pros or cons contain stopword phrases.
    stopword_file must contain columns: Phrase, Label, IsStopword
    """
    stop_df = pd.read_csv(stopword_file)
    stopwords = stop_df[stop_df["IsStopword"] == "Yes"]["Phrase"].str.lower().tolist()

    print(f"Loaded {len(stopwords)} stopwords to filter out.")

    def contains_stopword(text):
        t = str(text).lower()
        return any(sw in t for sw in stopwords)

    before = len(df)
    df = df[~df["pros"].apply(contains_stopword)]
    df = df[~df["cons"].apply(contains_stopword)]
    df = df.reset_index(drop=True)
    after = len(df)

    print(f"Removed {before - after} rows that match stopword phrases.")

    return df
# here it should be cleaned

def generate_topic_summary(text: str) -> str:
    """
    Classifies the combined pros/cons text into a single Topic Summary using Llama.
    """
    if not isinstance(text, str) or not text.strip():
        return "No Content"

    # Changed prompt to reflect 'Topic Summary' of work experience
    prompt = f"""
Analyze the following text, which combines pros and cons of a job experience, and summarize the core sentiment or main issue into ONE clear Topic Summary.
Rules:
- Output ONLY the summary name.
- Must be 5–10 words.
- No punctuation, no extra text.
- Don't focus on HR related terms. 
- Topics should be related to marketing (product/ brand advocacy).

Combined Pros and Cons: {text.strip()}

Topic Summary:
"""

    raw = run_llama_local(prompt)
    if not raw:
        return "Unknown Error"

    # Robust cleanup of Llama's output
    # Tries to find 'Topic Summary:' and clean the first line
    lines = raw.split("Topic Summary:", 1)[-1] if "Topic Summary:" in raw else raw
    topic = lines.splitlines()[0].strip().strip('."\'')
    
    words = topic.split()
    # Optional: Basic check to guide the model towards the desired length
    if 5 <= len(words) <= 10:
        return topic
    else:
        # Returns the topic even if length is wrong, allowing you to review model adherence
        return topic 

def main(excel_path: str, 
         sheet_name: str = 0,
         output_path: str = OUTPUT_FILE,
         max_rows: int = None):
    
    print(f"Loading: {excel_path}")
    
    df = pd.read_csv(excel_path)
    df = filter_by_year(df, 2021)
    df = remove_stopword_rows(df, STOPWORD_FILE)


    # Define the columns we need
    PROS_COL = "pros"
    CONS_COL = "cons"
    
    # --- Data Validation and Preprocessing ---
    required_cols = [PROS_COL, CONS_COL]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns! Not found: {missing}. Found: {list(df.columns)}")

    # Fill NaN values with an empty string before concatenation
    df[PROS_COL] = df[PROS_COL].fillna("")
    df[CONS_COL] = df[CONS_COL].fillna("")
    
    # 🌟 CORE CHANGE: Create a single text column by combining Pros and Cons
    df["combined_text"] = (
        "Pros: " + df[PROS_COL].astype(str).str.strip() + 
        " | Cons: " + df[CONS_COL].astype(str).str.strip()
    )
    
    # Filter out rows where both pros and cons were empty
    df = df[df["combined_text"].str.len() > 10].reset_index(drop=True)

    if max_rows:
        df = df.head(max_rows)

    print(f"Classifying {len(df)} rows based on combined 'pros' and 'cons'...")
    tqdm.pandas(desc="Generating Topic Summary")

    # Apply the new classification function to the combined text
    df["llama_topic_summary"] = df["combined_text"].progress_apply(generate_topic_summary)

    # --- Output ---
    # Include original columns, the combined text, and the Llama prediction
    result_df = df[[PROS_COL, CONS_COL, "combined_text", "llama_topic_summary"]]

    print(f"Saving {len(result_df)} rows to: {output_path}")
    result_df.to_excel(output_path, index=False)
    print("Done.")

    return result_df


if __name__ == "__main__":
    
    # NOTE: Set the maximum number of rows to process (e.g., for testing)
    # ROW_LIMIT = 15000 
    ROW_LIMIT = None


    try:
        main(
            excel_path=EXCEL_FILE,
            sheet_name=0,
            output_path=OUTPUT_FILE,
            max_rows=ROW_LIMIT
        )
    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found at {EXCEL_FILE}. Please check the path and file name.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

