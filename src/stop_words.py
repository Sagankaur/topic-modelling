def export_stopword_csv(excel_path, output_csv_path):
    """
    Export Phrase / Label / IsStopword into a CSV
    so llama_pros.py can load it.
    """
    all_sheets = pd.read_excel(excel_path, sheet_name=None)

    combined = []
    for sheet_name, df in all_sheets.items():
        if not {"Label", "IsStopword"}.issubset(df.columns):
            continue
        
        # First column is always the phrase
        phrase_col = df.columns[0]

        temp = df[[phrase_col, "Label", "IsStopword"]].copy()
        temp.rename(columns={phrase_col: "Phrase"}, inplace=True)
        combined.append(temp)

    if not combined:
        print("❌ No valid sheets found for CSV export.")
        return

    final_df = pd.concat(combined, ignore_index=True)
    final_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\n✅ Exported stopword CSV to: {output_csv_path}")
csv_path = r"D:\DOMS job reviews\classified_stopwords.csv"
excel_path=r"D:\DOMS job reviews\top_phrases.xlsx"
export_stopword_csv(excel_path, csv_path)