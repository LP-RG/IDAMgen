#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from pathlib import Path
import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = os.path.join(ROOT_DIR, "src")
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import models.resnet8 as resnet8
import models.resnet20 as resnet20
import modules.convolution as conv  

# Registry mapping --model_name to its model class; extend here for new architectures.
MODEL_REGISTRY = {
    "resnet8": resnet8.ResNet8,
    "resnet20": resnet20.ResNet20,
}
MODEL_KWARGS = dict(num_classes=10, conv_type=3, bit_width=8, signed=False, zone=False)

BASELINE_FOLDER = "baseline_test"
ID_PATTERN = re.compile(r"(\d{8,})_(\d+)")


def extract_id(name: str):
    m = ID_PATTERN.search(name)
    return (m.group(1), m.group(2)) if m else None


# Instantiate a throwaway (no-approximation) model and read back the layer
# names its Conv2d_custom submodules were built with, in definition order.
def get_layer_names_from_model(model_cls, **model_kwargs):
    dummy_model = model_cls(multiplier_matrix=None, **model_kwargs)
    layer_names = []
    for m in dummy_model.modules():
        if isinstance(m, conv.Conv2d_custom):
            layer_names.append(m.name)
    del dummy_model
    return layer_names


# Returns the non-dominated rows of df: minimize area, maximize accuracy_avg.
def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    areas = df["area"].values
    accs = df["accuracy_avg"].values
    is_pareto = [True] * len(df)

    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        for j in range(len(df)):
            if i == j:
                continue
            dominates = (
                areas[j] <= areas[i]
                and accs[j] >= accs[i]
                and (areas[j] < areas[i] or accs[j] > accs[i])
            )
            if dominates:
                is_pareto[i] = False
                break

    return df[pd.Series(is_pareto)].reset_index(drop=True)


# Peels successive pareto fronts from df (non-dominated sorting) until at
# least min_count rows are collected. Layer 0 is the real pareto front;
# layers 1, 2, ... are the next-best backup fronts. Adds a 'pareto_layer' column.
def compute_pareto_layers(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    remaining = df.reset_index(drop=True).copy()
    layers = []
    layer_num = 0
    collected = 0

    while len(remaining) > 0 and collected < min_count:
        front = compute_pareto_front(remaining)
        front = front.copy()
        front["pareto_layer"] = layer_num
        layers.append(front)
        collected += len(front)
        remaining = remaining[~remaining["file"].isin(front["file"])].reset_index(drop=True)
        layer_num += 1

    if not layers:
        empty = df.iloc[0:0].copy()
        empty["pareto_layer"] = pd.Series(dtype="int64")
        return empty

    return pd.concat(layers, ignore_index=True)


# Picks n rows from layered_df, evenly spaced by accuracy_avg (including
# both extremes). Returns all rows if there are fewer than n available.
def select_n_points(layered_df: pd.DataFrame, n: int) -> pd.DataFrame:
    sorted_df = layered_df.sort_values("accuracy_avg").reset_index(drop=True)
    total = len(sorted_df)

    if total <= n:
        return sorted_df
    if n <= 1:
        return sorted_df.iloc[[total // 2]].reset_index(drop=True)

    indices = sorted(set(round(i * (total - 1) / (n - 1)) for i in range(n)))
    if len(indices) < n:
        used = set(indices)
        rest = sorted(
            (i for i in range(total) if i not in used),
            key=lambda i: min(abs(i - idx) for idx in indices),
        )
        for i in rest:
            if len(indices) >= n:
                break
            indices.append(i)
        indices = sorted(indices)

    return sorted_df.loc[indices].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=sorted(MODEL_REGISTRY.keys()),
                         help="Model architecture; determines the per-layer output folders to create.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-multipliers", "-n", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)

    required_cols = ["file", "area", "accuracy", "accuracy_seed_100", "accuracy_seed_0"]
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=required_cols).copy()

    if df.empty:
        print("No complete rows available.")
        return

    df["accuracy_avg"] = df[["accuracy", "accuracy_seed_100", "accuracy_seed_0"]].mean(axis=1)

    layered_df = compute_pareto_layers(df, args.num_multipliers)
    selected_df = select_n_points(layered_df, args.num_multipliers)

    print(f"Selected multipliers ({len(selected_df)}/{args.num_multipliers}):")
    print(selected_df[["file", "area", "accuracy_avg", "pareto_layer"]].to_string(index=False))

    npy_by_id = {}
    for f in npy_dir.glob("*.npy"):
        fid = extract_id(f.stem)
        if fid is not None:
            npy_by_id.setdefault(fid, []).append(f)

    matched_files = []
    for _, row in selected_df.iterrows():
        cid = extract_id(str(row["file"]))
        matched_files.extend(npy_by_id.get(cid, []))

    print(f".npy files to copy into each folder: {len(matched_files)}")

    layer_names = get_layer_names_from_model(MODEL_REGISTRY[args.model_name], **MODEL_KWARGS)
    print(f"Layers found for '{args.model_name}': {layer_names}")

    target_folders = layer_names + [BASELINE_FOLDER]
    for folder_name in target_folders:
        target_dir = out_dir / folder_name
        if not args.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for src in matched_files:
            dst = target_dir / f"{src.stem}_{folder_name}{src.suffix}"
            if args.dry_run:
                print(f"[DRY-RUN] {src} -> {dst}")
            else:
                shutil.copy2(src, dst)


if __name__ == "__main__":
    main()