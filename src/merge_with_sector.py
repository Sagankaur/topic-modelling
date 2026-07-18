import pandas as pd
import glob

# --- Load files ---
classify = pd.read_excel("classify.xlsx", sheet_name="Company")
glassdoor = pd.read_csv("updated_glassdoor_data.csv")

# --- Standardize names ---
classify['Company'] = classify['Company'].str.strip().str.lower()
glassdoor['firm_name'] = glassdoor['firm_name'].str.strip().str.lower()

# --- Extract year ---
glassdoor['date'] = pd.to_datetime(glassdoor['date'], errors='coerce')
glassdoor['year'] = glassdoor['date'].dt.year

# --- Merge ---
merged = pd.merge(
    glassdoor,
    classify,
    left_on='firm_name',
    right_on='Company',
    how='left'
)

# --- Filter ---
'''
Manufacturing / Industrial / Engineering / Automation	4109
IT / Technology / Software	3665
Construction / Real Estate / Architecture	2961
Healthcare / Pharma / Biotech / Medical Devices	2640
Food & Beverage / Hospitality / Restaurants / Tourism	2257
Finance / Banking / Insurance / FinTech	2205
Education / EdTech / Research / Nonprofit	2180
Entertainment / Media / Arts / Sports / Gaming	2158
Government / Public Sector / Defense / Aerospace	2035
Retail / Consumer Goods / Fashion / E-Commerce	1888
Consulting / Professional Services / Outsourcing / BPO	1877
Transportation / Logistics / Automotive / Aviation / Maritime	1701
Energy / Utilities / Oil & Gas / Mining	1106
Unknown / Other	843
Telecom / Networking / Internet Services	716
Agriculture / Materials / Chemicals / FMCG	594
Advertising / Marketing / PR / Creative Agencies	431
Human Resources / Staffing / Recruitment	203
Cybersecurity / Information Security	187
Legal / Compliance / Law / Regulatory	175
Semiconductors / Electronics / Hardware	123
Environmental / Sustainability / CleanTech	32
Venture Capital / Private Equity / Investments	29
Media / Publishing / Journalism	28
Consumer Services	14
'''
year=2021
# sector_to_filter = "Energy / Utilities / Oil & Gas / Mining" #totlal rows =29,293
# "IT / Technology / Software" #total rows: 2,46,652
sector_to_filter ="IT / Technology / Software" #total rows: 2,46,652
filtered = merged[
    (merged['year'] == year) 
    &
    (merged['sector'] == sector_to_filter)
]

# --- Print head and row count ---
print("Filtered Data (Head):")
print(filtered.head())
print("\nNumber of rows before chunking:", len(filtered))

# --- PARTIAL / CHUNKED SAVING ---
chunk_size = 5000  # adjust as needed

for i in range(0, len(filtered), chunk_size):
    chunk = filtered.iloc[i:i + chunk_size]
    chunk.to_csv(f"filter_sector_part_{i//chunk_size + 1}.csv", index=False)

print("\nSaved in chunks successfully!")

# --- RECOMBINE CHUNKS INTO ONE FINAL FILE ---
files = sorted(glob.glob("filter_sector_part_*.csv"))
df_full = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

# --- Create final filename ---
sector_clean = sector_to_filter.replace(" ", "_").replace("/", "_")
final_filename = f"filtered_sector_{year}_{sector_clean}.csv"

df_full.to_csv(final_filename, index=False)

# --- Print final info ---
print(f"\nFinal merged file saved as: {final_filename}")
print("Final dataset head:")
print(df_full.head())
print("\nFinal total rows:", len(df_full))
