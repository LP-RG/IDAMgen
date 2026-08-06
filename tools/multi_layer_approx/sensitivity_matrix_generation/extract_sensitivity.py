import sys

import torch
import numpy as np
import torch.nn as nn
import os
import time
import argparse
import glob
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = os.path.join(ROOT_DIR, "src")
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import src.models.resnet8 as resnet8
import src.modules.data_loaders as data_loader

from src.cnn_training import calibration


trained_models_path = os.path.join(ROOT_DIR, "trained_models/")
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_forward_only(model, train_loader, criterion):
    """Do a single forward pass on a frozen batch to extract the loss."""

    model.eval()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    return loss.item()

def get_calibration_loader(model_name="resnet8", calib_batch_size=64, calib_samples=1024):
    """Build a fixed calibration loader: calib_samples total samples, calib_batch_size per batch."""
    print(f"Extracting Calibration Set ({calib_samples} samples, batch_size={calib_batch_size})...")
    calib_loader, _, _ = data_loader.get_datasets(calib_batch_size, model_name)
    num_batches = calib_samples // calib_batch_size
    return calib_loader, num_batches

def get_training_loader(batch_size, model_name="resnet8"):
    """Build the loader used for the forward-only loss evaluation.
    If batch_size is None, the entire training set is used as a single batch."""
    if batch_size is not None:
        print(f"Building training data loader (batch_size={batch_size})...")
        train_loader, _, _ = data_loader.get_datasets(batch_size, model_name)
        return train_loader

    print("No --batch_size specified: using the entire training set as a single batch.")
    probe_loader, _, _ = data_loader.get_datasets(1, model_name)
    full_size = len(probe_loader.dataset)
    train_loader, _, _ = data_loader.get_datasets(full_size, model_name)
    return train_loader

def calibration(model, calib_loader, num_batches, stats=False):
    """Do forward passes over `num_batches` batches for calibration."""
    print("Calibrating model for activation scales...")
    if stats:
        model.eval()
    else:
        model.train()

    for i, (inputs, _) in enumerate(calib_loader):
        if i >= num_batches:
            break
        inputs = inputs.to(device)
        model(inputs)

def main():
    parser = argparse.ArgumentParser(description="Extract sensitivity matrix for ResNet8 using forward pass.")
    
    parser.add_argument("--1_1_approx", type=str, default=None, help="Path to folder with .npy files for Layer 1")
    parser.add_argument("--1_2_approx", type=str, default=None, help="Path to folder with .npy files for Layer 2")
    parser.add_argument("--2_1_approx", type=str, default=None, help="Path to folder with .npy files for Layer 3")
    parser.add_argument("--2_2_approx", type=str, default=None, help="Path to folder with .npy files for Layer 4")
    parser.add_argument("--2_s_approx", type=str, default=None, help="Path to folder with .npy files for Layer 5")
    parser.add_argument("--3_1_approx", type=str, default=None, help="Path to folder with .npy files for Layer 6")
    parser.add_argument("--3_2_approx", type=str, default=None, help="Path to folder with .npy files for Layer 7")

    
    parser.add_argument("--batch_size", type=int, default=None, help="Size of the batches used for forward-only evaluation. If omitted, the whole training set is used as a single batch.")
    parser.add_argument("--quant_model_path", type=str, default=os.path.join(ROOT_DIR, "trained_models/resnet8_q8.pth"), help="Path to the quantized model")
    args = parser.parse_args()
    
    if not os.path.exists(args.quant_model_path):
        raise FileNotFoundError(f"Quantized model not found in {args.quant_model_path}. Please run the quantized training first.")

    args_dict = vars(args)
    layer_dirs = [
        args_dict['1_1_approx'], 
        args_dict['1_2_approx'], 
        args_dict['2_1_approx'], 
        args_dict['2_2_approx'], 
        args_dict['2_s_approx'], 
        args_dict['3_1_approx'], 
        args_dict['3_2_approx']
    ]
    
    layer_candidates = [] 
    global_index_map = []
    
    for layer_idx, l_dir in enumerate(layer_dirs):
        if l_dir is not None and os.path.exists(l_dir):
            files = sorted(glob.glob(os.path.join(l_dir, "*.npy")))
            layer_candidates.append(files)
            for f in files:
                global_index_map.append({
                    "layer": layer_idx + 1,
                    "file": os.path.basename(f),
                    "path": f
                })
        else:
            layer_candidates.append([])
            
    total_candidates = sum(len(cands) for cands in layer_candidates)
    if total_candidates == 0:
        raise ValueError("No .npy files found in the specified folders.")

    print(f"Initializing matrix {total_candidates}x{total_candidates}...")
    S_matrix = np.zeros((total_candidates, total_candidates))
    
    def get_g_idx(layer_i, file_path):
        for idx, item in enumerate(global_index_map):
            if item["path"] == file_path:
                return idx
        return -1

    calib_loader, calib_num_batches = get_calibration_loader(model_name="resnet8", calib_batch_size=64, calib_samples=1024)
    train_loader = get_training_loader(args.batch_size, "resnet8")
    criterion = nn.CrossEntropyLoss()
    quant_state_dict = torch.load(args.quant_model_path, weights_only=True)

    def instantiate_and_eval(multiplier_files):
        """Instantiate ResNet8, inject the specified .npy files, load the weights and calculate the loss."""

        setup_seed(42)

        model = resnet8.ResNet8(
            multiplier_matrix=multiplier_files, 
            num_classes=10, 
            conv_type=3,
            bit_width=8, 
            signed=False, 
            zone=False
        ).to(device)
        model.load_state_dict(quant_state_dict)

        calibration(model, calib_loader, calib_num_batches)

        return evaluate_forward_only(model, train_loader, criterion)

    print("Calcolo Loss Baseline (Quantized)...")
    baseline_loss = instantiate_and_eval(None)
    print(f"Baseline Loss: {baseline_loss:.6f}")

    start_time = time.time()

    # Diagonal (individual sensitivities) and off-diagonal (cross-layer interactions) extraction
    print("\n--- Extraction of Individual Sensitivities ---")
    for l_idx, files in enumerate(layer_candidates):
        for f in files:
            loss = instantiate_and_eval([f])
            delta_L = loss - baseline_loss
            
            g_idx = get_g_idx(l_idx, f)
            S_matrix[g_idx, g_idx] = delta_L
            print(f"Layer {l_idx+1} | File: {os.path.basename(f)} | Delta: {delta_L:.6f}")

    print("\n--- Extraction of Cross-Layer Interactions ---")
    active_layers = [i for i, cands in enumerate(layer_candidates) if len(cands) > 0]
    
    for i in range(len(active_layers)):
        for j in range(i + 1, len(active_layers)):
            l1_idx = active_layers[i]
            l2_idx = active_layers[j]
            
            for f1 in layer_candidates[l1_idx]:
                for f2 in layer_candidates[l2_idx]:
                    
                    loss_combinata = instantiate_and_eval([f1, f2])
                    
                    g1_idx = get_g_idx(l1_idx, f1)
                    g2_idx = get_g_idx(l2_idx, f2)
                    
                    delta1 = S_matrix[g1_idx, g1_idx]
                    delta2 = S_matrix[g2_idx, g2_idx]
                    
                    interazione = ((loss_combinata - baseline_loss) - delta1 - delta2) / 2
                    
                    S_matrix[g1_idx, g2_idx] = interazione
                    S_matrix[g2_idx, g1_idx] = interazione
                    
                    print(f"Pair (L{l1_idx+1}, L{l2_idx+1}) | M1: {os.path.basename(f1)}, M2: {os.path.basename(f2)} | Inter: {interazione:.6f}")

    print(f"\nExtraction completed in {time.time() - start_time:.2f} seconds.")

    result_folder_path = os.path.join(CURR_DIR, "sensitivity_matrices")
    os.makedirs(result_folder_path, exist_ok=True)

    experiment_timestamp = int(time.time())

    experiment_folder_path = os.path.join(CURR_DIR, "sensitivity_matrices", f"resnet8_sensitivity_{experiment_timestamp}")
    os.makedirs(experiment_folder_path, exist_ok=True)

    experiment_name = f"resnet8_sensitivity_matrix_{experiment_timestamp}"

    out_matrix_path = os.path.join(experiment_folder_path, f"{experiment_name}.npy")
    out_map_path = os.path.join(experiment_folder_path, f"{experiment_name}_map.json")
    
    np.save(out_matrix_path, S_matrix)
    with open(out_map_path, 'w') as map_file:
        json.dump(global_index_map, map_file, indent=4)
        
    print(f"Matrix saved in: {out_matrix_path}")
    print(f"Index mapping saved in: {out_map_path}")

if __name__ == "__main__":
    main()