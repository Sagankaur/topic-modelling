import pandas as pd
import subprocess
import time
import os

# -------------------------
# CONFIG
# -------------------------
EXCEL_PATH = r"D:\DOMS job reviews\classify.xlsx"
SHEET_NAME = "Company"
OUTPUT_COLUMN = "sector"
MODEL = "llama3.1:8b"

SAVE_EVERY = 20

SECTORS = [
    "IT / Technology / Software",
    "Telecom / Networking / Internet Services",
    "Entertainment / Media / Arts / Sports / Gaming",
    "Food & Beverage / Hospitality / Restaurants / Tourism",
    "Manufacturing / Industrial / Engineering / Automation",
    "Retail / Consumer Goods / Fashion / E-Commerce",
    "Healthcare / Pharma / Biotech / Medical Devices",
    "Finance / Banking / Insurance / FinTech",
    "Consulting / Professional Services / Outsourcing / BPO",
    "Government / Public Sector / Defense / Aerospace",
    "Construction / Real Estate / Architecture",
    "Energy / Utilities / Oil & Gas / Mining",
    "Transportation / Logistics / Automotive / Aviation / Maritime",
    "Legal / Compliance / Law / Regulatory",
    "Agriculture / Materials / Chemicals / FMCG",
    "Education / EdTech / Research / Nonprofit",
    "Human Resources / Staffing / Recruitment",
    "Advertising / Marketing / PR / Creative Agencies",
    "Cybersecurity / Information Security",
    "Semiconductors / Electronics / Hardware",
    "Venture Capital / Private Equity / Investments",
    "Environmental / Sustainability / CleanTech",
    "Consumer Services",
    "Media / Publishing / Journalism",
    "Unknown / Other"
]

# -------------------------
# OLLAMA CALL
# -------------------------
def classify_company(name):
    prompt = f"""
Classify the company below into EXACTLY ONE of these sectors:

{SECTORS}

Company: "{name}"

ONLY return the sector name. No explanation.
"""
    try:
        proc = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = proc.stdout.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        print("❌ Error calling Ollama:", e)
        return "Unknown / Other"

    # match sector
    output_l = output.lower()
    for sec in SECTORS:
        if sec.lower() in output_l:
            return sec

    return "Unknown / Other"


# -------------------------
# SAFE SAVE
# -------------------------
def save_partial(df):
    temp_path = EXCEL_PATH.replace(".xlsx", "_partial.xlsx")
    df.to_excel(temp_path, sheet_name=SHEET_NAME, index=False)
    os.replace(temp_path, EXCEL_PATH)
    print("💾 Partial save complete.")


# -------------------------
# LOAD SHEET
# -------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Create column if missing
if OUTPUT_COLUMN not in df.columns:
    df[OUTPUT_COLUMN] = ""

# -------------------------
# FIND FIRST UNCLASSIFIED ROW
# -------------------------
unclassified = df[df[OUTPUT_COLUMN].isna() | (df[OUTPUT_COLUMN] == "")]
if len(unclassified) == 0:
    print("🎉 All rows are already classified. Nothing to do!")
    exit()

START_ROW = unclassified.index[0]

print(f"🔎 Detected first unclassified row at index: {START_ROW}")
print(f"➡ Continuing classification from row {START_ROW + 1}...\n")

# -------------------------
# MAIN LOOP
# -------------------------
for i in range(START_ROW, len(df)):
    company = str(df.at[i, "Company"])
    print(f"{i+1}/{len(df)} → {company}")

    sector = classify_company(company)
    df.at[i, OUTPUT_COLUMN] = sector

    print("   Sector →", sector)

    # save after each classification
    save_partial(df)

    # also save every N rows
    if (i + 1) % SAVE_EVERY == 0:
        save_partial(df)

    time.sleep(0.05)

# FINAL SAVE
save_partial(df)
print("✔ Done! All remaining companies classified.")
