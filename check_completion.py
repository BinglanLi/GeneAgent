"""
check_completion.py
Regenerates incomplete_by_part.csv by checking each (LLM, part, noise_level)
combination in Outputs/ against its input file, then moves any newly-complete
noise-level folders to Outputs_Finished/.

Usage:
    python check_completion.py
"""

import os
import shutil
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS     = os.path.join(BASE_DIR, "Outputs")
FINISHED    = os.path.join(BASE_DIR, "Outputs_Finished")
INPUT_DIR   = os.path.join(BASE_DIR, "Datasets", "AlzKB")
REPORT_PATH = os.path.join(OUTPUTS, "incomplete_by_part.csv")

LLMS         = ["gpt-oss:20b", "gemma4:26b", "mixtral:8x22b"]
NOISE_LEVELS = ["full_set", "reduced_set", "noise_20", "noise_40", "noise_60", "noise_80"]


def load_output_csv(path):
    """Return DataFrame or None if file is missing/empty."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except pd.errors.EmptyDataError:
        return None


def check_all():
    incomplete_rows = []
    newly_finished  = []

    for llm in LLMS:
        llm_dir = os.path.join(OUTPUTS, llm)
        if not os.path.isdir(llm_dir):
            continue

        for part_folder in sorted(os.listdir(llm_dir)):
            if not part_folder.startswith("sampled_noise_part"):
                continue

            part_num   = part_folder.replace("sampled_noise_", "")   # e.g. "part3"
            input_path = os.path.join(INPUT_DIR, f"{part_folder}.csv")
            if not os.path.exists(input_path):
                continue

            df_in        = pd.read_csv(input_path)
            expected     = len(df_in)
            expected_ids = set(range(expected))

            eval_csv = os.path.join(OUTPUTS, llm, part_folder,
                                    "evaluation_results_processNames.csv")
            df_out = load_output_csv(eval_csv)

            for noise in NOISE_LEVELS:
                noise_dir = os.path.join(OUTPUTS, llm, part_folder, noise)
                if not os.path.isdir(noise_dir):
                    continue  # already moved or never existed

                if df_out is not None:
                    done_ids  = set(df_out.loc[df_out["prediction_type"] == noise,
                                               "pathway_id"])
                    completed = len(expected_ids & done_ids)
                else:
                    completed = 0

                missing = expected - completed

                if missing > 0:
                    incomplete_rows.append({
                        "input_file":  f"{part_folder}.csv",
                        "llm":         llm,
                        "noise_level": noise,
                        "part":        part_num,
                        "expected":    expected,
                        "completed":   completed,
                        "missing":     missing,
                    })
                else:
                    newly_finished.append((llm, part_folder, noise, noise_dir))

    # Write updated report
    df_report = pd.DataFrame(incomplete_rows)
    df_report.to_csv(REPORT_PATH, index=False)
    print(f"Wrote {REPORT_PATH}  ({len(df_report)} incomplete combinations)")

    # Move finished folders
    for llm, part_folder, noise, src in newly_finished:
        dst = os.path.join(FINISHED, llm, part_folder, noise)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"  Moved to Outputs_Finished: {llm}/{part_folder}/{noise}")

    print(f"\nDone. {len(newly_finished)} folder(s) moved to Outputs_Finished/.")


if __name__ == "__main__":
    check_all()
