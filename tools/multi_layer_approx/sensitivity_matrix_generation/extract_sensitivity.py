import sys
import torch
import numpy as np
import torch.nn as nn
import os
import time
import argparse
import glob
import json
import mat_mul
import csv      
import shutil   

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = os.path.join(ROOT_DIR, "src")
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import src.models.resnet8 as resnet8
import src.modules.data_loaders as data_loader
import src.modules.convolution as conv

trained_models_path = os.path.join(ROOT_DIR, "trained_models/")
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_forward_only(model, train_loader, criterion, num_batches=None):
    debug_batch_idx = 4
    total_loss = 0.0
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(train_loader):
            if num_batches is not None and batch >= num_batches:
                break

            if batch == debug_batch_idx:
                # DEBUG PRINT JUST TO BE SURE THAT THE IPUTS ARE ALLWAYS THE SAME
                pixel_sums = inputs.view(inputs.size(0), -1).sum(dim=1)
                print(f"[DEBUG] Batch {batch} | somma pixel per immagine: "
                      f"{[f'{v:.4f}' for v in pixel_sums.tolist()]}")
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def pre_training(model, train_loader, criterion, optimizer, epoch=5):
    model.train()
    for _ in range(epoch):
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)            
            loss.backward()


def set_data_loaders(model_name: str):
    """Sets appropriate batch sizes based on the model architecture and loads data."""
    global train_loader, test_loader, _classes, batch_size
    train_loader, test_loader, _classes = data_loader.get_datasets(BATCH_SIZE, model_name)


def calibration(model):
    """Calibrates model activations/weights using the training set."""
    print("Calibrating model...")

    for m in model.modules():
        if isinstance(m, conv.Conv2d_custom):
            m.calibrating = True

    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= 1024 // BATCH_SIZE:
                break
            inputs = inputs.to(device)
            model(inputs)

    for m in model.modules():
        if isinstance(m, conv.Conv2d_custom):
            m.freeze_qparams()

def main():
    parser = argparse.ArgumentParser(description="Extract sensitivity matrix for ResNet8 using forward pass.")
    
    parser.add_argument("--1_1_approx", type=str, default=None, help="Path to folder with .npy files for Layer 1")
    parser.add_argument("--1_2_approx", type=str, default=None, help="Path to folder with .npy files for Layer 2")
    parser.add_argument("--2_1_approx", type=str, default=None, help="Path to folder with .npy files for Layer 3")
    parser.add_argument("--2_2_approx", type=str, default=None, help="Path to folder with .npy files for Layer 4")
    parser.add_argument("--2_s_approx", type=str, default=None, help="Path to folder with .npy files for Layer 5")
    parser.add_argument("--3_1_approx", type=str, default=None, help="Path to folder with .npy files for Layer 6")
    parser.add_argument("--3_2_approx", type=str, default=None, help="Path to folder with .npy files for Layer 7")

    parser.add_argument("--full_test", action="store_true", help="If set, evaluate on the full test set.")
    parser.add_argument("--pre_training", action="store_true", help="If set, train the model with pre_training before evaluating on the test set.")
    parser.add_argument("--batch_number", type=int, default=None, help="Number of batches (fixed batch_size=64) to see before returning the loss. If omitted, iterates over the whole training set.")
    parser.add_argument("--quant_model_path", type=str, default=os.path.join(ROOT_DIR, "trained_models/resnet8_q8.pth"), help="Path to the quantized model")
    
    
    parser.add_argument("--csv_path", type=str, default="results.csv", help="Path to the results CSV file")
    args = parser.parse_args()
    
    if not os.path.exists(args.quant_model_path):
        raise FileNotFoundError(f"Quantized model not found in {args.quant_model_path}. Please run the quantized training first.")

    csv_metrics = {}
    if os.path.exists(args.csv_path):
        with open(args.csv_path, mode='r') as f_csv:
            reader = csv.DictReader(f_csv)
            for row in reader:
                csv_metrics[row['file']] = row
    else:
        print(f"Warning: CSV file '{args.csv_path}' not found.")

    args_dict = vars(args)
    
    layer_flags = ['1_1_approx', '1_2_approx', '2_1_approx', '2_2_approx', '2_s_approx', '3_1_approx', '3_2_approx']
    
    layer_candidates = [] 
    global_index_map = []
    
    for layer_idx, flag_name in enumerate(layer_flags):
        l_dir = args_dict[flag_name]
        if l_dir is not None and os.path.exists(l_dir):
            
            layer_str = flag_name.replace('_approx', '')
            layer_num = layer_idx + 1
            
            renamed_dir = os.path.join(l_dir, "renamed_multipliers")
            os.makedirs(renamed_dir, exist_ok=True)
            
            files = sorted(glob.glob(os.path.join(l_dir, "*.npy")))
            renamed_files_for_layer = []
            
            for f in files:
                base_name = os.path.basename(f)
                name_no_ext = os.path.splitext(base_name)[0]

                temp_name_no_ext = name_no_ext
                
                for suffix in ['_1_1', '_1_2', '_2_1', '_2_2', '_2_s', '_3_1', '_3_2']:
                    if temp_name_no_ext.endswith(suffix):
                        temp_name_no_ext = temp_name_no_ext[:-len(suffix)]
                        break
                
                new_name = f"{temp_name_no_ext}_{layer_str}.npy"

                new_path = os.path.join(renamed_dir, new_name)
                shutil.copy(f, new_path)
                renamed_files_for_layer.append(new_path)
                
                entry = {
                    "layer_idx": layer_num,
                    "layer": layer_str,
                    "file": new_name,
                    "original_file": base_name,
                    "path_original_file": f
                }
                
                if name_no_ext in csv_metrics:
                    metrics = csv_metrics[name_no_ext]
                    for key in ['area', 'power', 'delay', 'pda', 'mean_ae', 'mean_ae_cnn', 'max_ae', 'accuracy']:
                        if key in metrics:
                            entry[key] = float(metrics[key])
                    
                    if 'power' in entry and 'area' in entry:
                        entry['pa'] = entry['power'] * entry['area']
                
                global_index_map.append(entry)
            
            layer_candidates.append(renamed_files_for_layer)
        else:
            layer_candidates.append([])
            
    total_candidates = sum(len(cands) for cands in layer_candidates)
    if total_candidates == 0:
        raise ValueError("No .npy files found in the specified folders.")

    print(f"Initializing matrix {total_candidates}x{total_candidates}...")
    S_matrix = np.zeros((total_candidates, total_candidates))
    
    def get_g_idx(layer_i, file_path):
        basename = os.path.basename(file_path)
        for idx, item in enumerate(global_index_map):
            if item["file"] == basename and item["layer_idx"] == layer_i + 1:
                return idx
        return -1

    criterion = nn.CrossEntropyLoss()

    def instantiate_and_eval(multiplier_files, baseline=False):
        """Instantiate ResNet8, inject the specified .npy files, load the weights and calculate the loss."""
        quant_state_dict = torch.load(args.quant_model_path, weights_only=True)
        setup_seed(42)
        set_data_loaders("resnet8")
        model = resnet8.ResNet8(
            multiplier_matrix=multiplier_files,
            num_classes=10,
            conv_type=3,
            bit_width=8,
            signed=False,
            zone=False,
        ).to(device)

        model.load_state_dict(quant_state_dict)
        calibration(model)

        if args.pre_training:
            if baseline:
                return evaluate_forward_only(model, test_loader, criterion)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
            #print(f"Pre-training starting. LOSS JUST FOR DEBUG : {evaluate_forward_only(model, train_loader, criterion)}")
            pre_training(model, train_loader, criterion, optimizer)
            #print(f"Pre-training completed. LOSS JUST FOR DEBUG : {evaluate_forward_only(model, train_loader, criterion)}")
            return evaluate_forward_only(model, test_loader, criterion)

        if baseline:
            if args.full_test:
                return evaluate_forward_only(model, test_loader, criterion)
            return evaluate_forward_only(model, train_loader, criterion)

        if args.full_test:
            return evaluate_forward_only(model, test_loader, criterion)

        return evaluate_forward_only(model, train_loader, criterion, num_batches=args.batch_number)

    print("Calcolo Loss Baseline (Quantized)...")
    #TODO: check if does make sense to use the loss over the entiry dataset
    baseline_loss = instantiate_and_eval(None, baseline=True)
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
    experiment_folder_path = os.path.join(CURR_DIR, "sensitivity_matrices", f"resnet8_sensitivity_n_samples_{'full_test' if args.full_test else str(BATCH_SIZE * args.batch_number if args.batch_number is not None else 1)}_{experiment_timestamp}")
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