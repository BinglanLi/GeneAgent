"""
check_completion.py
Regenerates incomplete_by_part.csv by checking each (LLM, part, noise_level)
combination in Outputs/ against its input file, then moves any newly-complete
noise-level folders to Outputs_Finished/, and rewrites run_geneagent_gpu_incomplete.slurm
with the current set of incomplete (LLM, part) tasks.

Usage:
    python check_completion.py
"""

import os
import re
import shutil
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS     = os.path.join(BASE_DIR, "Outputs")
FINISHED    = os.path.join(BASE_DIR, "Outputs_Finished")
INPUT_DIR   = os.path.join(BASE_DIR, "Datasets", "AlzKB")
REPORT_PATH = os.path.join(OUTPUTS, "incomplete_by_part.csv")

LLMS         = ["gpt-oss:20b", "gemma4:26b", "mixtral:8x22b"]
NOISE_LEVELS = ["full_set", "reduced_set", "noise_20", "noise_40", "noise_60", "noise_80"]
SLURM_PATH   = os.path.join(BASE_DIR, "run_geneagent_gpu_incomplete.slurm")


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

    generate_slurm(incomplete_rows)


def generate_slurm(incomplete_rows):
    """Rewrite the CONFIGS block and --array directive in the slurm script."""
    if not os.path.exists(SLURM_PATH):
        print(f"Slurm template not found: {SLURM_PATH}")
        return

    # Group incomplete noise levels by (llm, part)
    groups = {}
    for row in incomplete_rows:
        key = (row["llm"], row["part"])   # part is like "part3"
        groups.setdefault(key, []).append(row["noise_level"])

    if not groups:
        print("No incomplete combinations — slurm script not updated.")
        return

    # Sort by canonical LLM order, then numeric part
    llm_order = {llm: i for i, llm in enumerate(LLMS)}
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (llm_order.get(k[0], 999), int(k[1].replace("part", "")))
    )
    n_tasks = len(sorted_keys)

    # Compute per-LLM index ranges for the section comments
    llm_ranges = {}
    for idx, (llm, _) in enumerate(sorted_keys):
        if llm not in llm_ranges:
            llm_ranges[llm] = [idx, idx]
        else:
            llm_ranges[llm][1] = idx

    # Build CONFIGS lines with LLM section headers
    config_lines = []
    current_llm = None
    for idx, (llm, part) in enumerate(sorted_keys):
        if llm != current_llm:
            current_llm = llm
            s, e = llm_ranges[llm]
            index_str = f"index {s}" if s == e else f"indices {s}-{e}"
            config_lines.append(f"    # {llm} ({index_str})")
        part_num = part.replace("part", "")
        # preserve canonical noise-level ordering
        # noise_levels here is to be skipped
        noise_str = " ".join(n for n in NOISE_LEVELS if n not in groups[(llm, part)])
        config_lines.append(f'    "{llm}|{part_num}|{noise_str}"')

    configs_block = "CONFIGS=(\n" + "\n".join(config_lines) + "\n)"

    counts_str = " + ".join(
        f"{llm} ({llm_ranges[llm][1] - llm_ranges[llm][0] + 1})"
        for llm in LLMS if llm in llm_ranges
    )
    array_range = "0" if n_tasks == 1 else f"0-{n_tasks - 1}"
    array_line = f"#SBATCH --array={array_range}          # {n_tasks} tasks: {counts_str}"

    with open(SLURM_PATH) as f:
        content = f.read()

    content = re.sub(r"#SBATCH --array=\S+[^\n]*", array_line, content)
    content = re.sub(r"CONFIGS=\(.*?\n\)", configs_block, content, flags=re.DOTALL)

    with open(SLURM_PATH, "w") as f:
        f.write(content)

    print(f"Updated {SLURM_PATH}: {n_tasks} task(s) across "
          f"{len(llm_ranges)} LLM(s).")


if __name__ == "__main__":
    check_all()
