import json
import os
import shutil
import zipfile
from datetime import datetime

from src.config import CFG


def save_run_to_drive(model_key: str, local_results_dir: str, drive_base: str):
    """Copy all results for this model run to Drive (or any local destination)."""
    dest = f"{drive_base}/{model_key}"
    os.makedirs(dest, exist_ok=True)

    files_to_save = [
        f"{local_results_dir}/{model_key}_raw_responses.json",
        f"{local_results_dir}/{model_key}_evaluated.json",
        f"{local_results_dir}/{model_key}_adversarial_evaluated.json",
        f"{local_results_dir}/{model_key}_metrics.json",
        f"{local_results_dir}/{model_key}_full_report.json",
        f"{local_results_dir}/{model_key}_report.md",
    ]

    saved = []
    for fp in files_to_save:
        if os.path.exists(fp):
            shutil.copy2(fp, dest)
            saved.append(os.path.basename(fp))

    # Also copy plots if they exist
    plots_dir = f"{local_results_dir}/plots/{model_key}"
    if os.path.exists(plots_dir):
        dest_plots = f"{dest}/plots"
        if os.path.exists(dest_plots):
            shutil.rmtree(dest_plots)
        shutil.copytree(plots_dir, dest_plots)
        saved.append("plots/")

    print(f"\n💾 Saved {len(saved)} items to → {dest}")
    for s in saved:
        print(f"   {s}")
    return dest


def load_run_from_drive(model_key: str, drive_base: str, local_results_dir: str):
    """Load a previous model's results from Drive back into the local results dir."""
    src = f"{drive_base}/{model_key}"
    if not os.path.exists(src):
        print(f"❌ No results found for '{model_key}' at {src}")
        return None

    os.makedirs(local_results_dir, exist_ok=True)
    loaded = {}

    file_map = {
        "evaluated"           : f"{model_key}_evaluated.json",
        "adv_evaluated"       : f"{model_key}_adversarial_evaluated.json",
        "metrics"             : f"{model_key}_metrics.json",
        "full_report"         : f"{model_key}_full_report.json",
    }

    for key, fname in file_map.items():
        src_path = f"{src}/{fname}"
        if os.path.exists(src_path):
            dst_path = f"{local_results_dir}/{fname}"
            shutil.copy2(src_path, dst_path)
            with open(dst_path) as f:
                loaded[key] = json.load(f)
            print(f"✅ Loaded {fname}")
        else:
            print(f"⚠️  Not found: {fname}")

    print(f"\n📂 Results for '{model_key}' restored from {drive_base}")
    return loaded


def compare_models_from_drive(model_keys: list, drive_base: str):
    """Load metrics for multiple models and print a comparison table."""
    all_metrics = {}
    for mk in model_keys:
        path = f"{drive_base}/{mk}/{mk}_metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[mk] = json.load(f)
            print(f"✅ Loaded metrics for {mk}")
        else:
            print(f"⚠️  No metrics found for {mk}")

    if not all_metrics:
        print("No models to compare.")
        return

    keys = ["accuracy_score", "relevance_score", "safety_score",
            "robustness_score", "toxicity_rate", "final_weighted_score"]
    labels = ["Accuracy", "Relevance", "Safety", "Robustness", "Toxicity%", "★ Final"]

    col_w = 14
    header = f"{'Metric':<20}" + "".join(f"{mk:>{col_w}}" for mk in all_metrics)
    print("\n" + "=" * (20 + col_w * len(all_metrics)))
    print("  MODEL COMPARISON")
    print("=" * (20 + col_w * len(all_metrics)))
    print(header)
    print("-" * (20 + col_w * len(all_metrics)))
    for k, label in zip(keys, labels):
        row = f"{label:<20}"
        for mk, m in all_metrics.items():
            val = m.get(k, "N/A")
            row += f"{str(val):>{col_w}}"
        print(row)
    print("=" * (20 + col_w * len(all_metrics)))
    return all_metrics


def zip_and_archive_to_drive(model_key: str, drive_base: str,
                              local_dirs: list = None, plot_paths: list = None):
    """
    Zips all evaluation outputs + plots into a dated archive.
    Folder structure:
        <drive_base>/
            archives/
                2025-01-15_14-30-00_qwen3b/
                    qwen3b_outputs.zip
    """
    if local_dirs is None:
        local_dirs = ["evaluation/results", "evaluation/logs", "data"]

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{model_key}"
    archive_dir = f"{drive_base}/archives/{folder_name}"
    os.makedirs(archive_dir, exist_ok=True)

    zip_path = f"{archive_dir}/{model_key}_outputs.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for local_dir in local_dirs:
            if not os.path.exists(local_dir):
                continue
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname   = os.path.relpath(full_path, start=".")
                    zf.write(full_path, arcname)

        # Also zip any loose plot PNGs passed in
        if plot_paths:
            for pp in plot_paths:
                if os.path.exists(pp):
                    zf.write(pp, os.path.join("plots", os.path.basename(pp)))

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)

    # Write a manifest file alongside the zip
    manifest = {
        "model_key"    : model_key,
        "timestamp"    : timestamp,
        "archived_dirs": local_dirs,
        "zip_size_mb"  : round(zip_size_mb, 2),
        "zip_path"     : zip_path
    }
    with open(f"{archive_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📦 Archive created: {zip_path}")
    print(f"   Size     : {zip_size_mb:.2f} MB")
    print(f"   Location : {archive_dir}")
    print(f"   Manifest : {archive_dir}/manifest.json")
    return zip_path