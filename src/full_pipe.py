import argparse
import subprocess
import os
import sys
import time

BASE_DIR = r"D:\DOMS job reviews"

# ----------------------------
# RUN A PYTHON SCRIPT
# ----------------------------
def run_script(script_name, args_list):
    print(f"\n==============================")
    print(f"▶ Running: {script_name}")
    print(f"==============================")

    cmd = ["python", script_name] + args_list
    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed! Stopping pipeline.")
        sys.exit(1)

    print(f"✔ Completed: {script_name}\n")
    time.sleep(1)


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Full Review → Clean → Topic Model Pipeline")

    parser.add_argument("--year", type=int, required=True, help="Filter reviews by year")
    parser.add_argument("--sector", type=str, required=True, help="Target job sector")

    parser.add_argument("--input_file", default=os.path.join(BASE_DIR, "updated_glassdoor_data.csv"))
    parser.add_argument("--merged_file", default=os.path.join(BASE_DIR, "merged_with_sector.csv"))
    parser.add_argument("--filtered_file", default=os.path.join(BASE_DIR, "filtered_sector.csv"))
    parser.add_argument("--cleaned_file", default=os.path.join(BASE_DIR, "cleaned_sector.csv"))
    parser.add_argument("--topic_output", default=os.path.join(BASE_DIR, "topics_output"))

    # dashboard and report
    parser.add_argument("--run_dashboard", action="store_true", help="Launch dashboard after pipeline")
    parser.add_argument("--run_report", action="store_true", help="Generate PDF report")

    args = parser.parse_args()

    print("\n========================================")
    print("      🚀 FULL TOPIC MODEL PIPELINE")
    print("========================================")
    print(f"Year:   {args.year}")
    print(f"Sector: {args.sector}")
    print(f"Dashboard: {'ON' if args.run_dashboard else 'OFF'}")
    print(f"Report:    {'ON' if args.run_report else 'OFF'}")
    print("========================================\n")

    # --------------------------------------
    # STEP 1: merge + sector filter
    # --------------------------------------
    run_script(
        "merge_with_sector.py",
        [
            "--year", str(args.year),
            "--sector", args.sector,
            "--output", args.filtered_file
        ]
    )

    # --------------------------------------
    # STEP 2: clean text
    # --------------------------------------
    run_script(
        "clean_data.py",
        [
            "--input", args.filtered_file,
            "--output", args.cleaned_file
        ]
    )

    # --------------------------------------
    # STEP 3: topic modeling
    # --------------------------------------
    run_script(
        "merge_clean_topic.py",
        [
            "--input", args.cleaned_file,
            "--output", args.topic_output
        ]
    )

    # --------------------------------------
    # STEP 4: Generate Report (PDF)
    # --------------------------------------
    if args.run_report:
        run_script(
            "generate_report.py",
            [
                "--input_dir", args.topic_output
            ]
        )

    # --------------------------------------
    # STEP 5: Launch Dashboard
    # --------------------------------------
    if args.run_dashboard:
        print("\nStarting Dashboard... (CTRL+C to stop)")
        run_script(
            "dashboard.py",
            [
                "--input_dir", args.topic_output
            ]
        )

    # --------------------------------------
    # DONE
    # --------------------------------------
    print("\n========================================")
    print("  🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("  Output folder →", args.topic_output)
    print("========================================\n")


if __name__ == "__main__":
    main()


# ✔ Run full pipeline with PDF report + dashboard
# python full_pipeline.py --year 2023 --sector "Energy / Utilities / Oil & Gas / Mining" --run_report --run_dashboard

# ✔ Run only data + model steps (no dashboard/report)
# python full_pipeline.py --year 2023 --sector "Energy / Utilities / Oil & Gas / Mining"

# ✔ Run + generate only PDF report
# python full_pipeline.py --year 2023 --sector "Energy / Utilities / Oil & Gas / Mining" --run_report

# ✔ Run + launch dashboard only
# python full_pipeline.py --year 2023 --sector "Energy / Utilities / Oil & Gas 