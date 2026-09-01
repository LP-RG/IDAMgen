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
import gc
import re 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = os.path.join(ROOT_DIR, "src")
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import src.models.resnet8 as resnet8
import src.models.resnet20 as resnet20
import src.modules.data_loaders as data_loader
import src.modules.convolution as conv

# Registro dei modelli supportati
MODEL_REGISTRY = {
    "resnet8": resnet8.ResNet8,
    "resnet20": resnet20.ResNet20,
}
MODEL_KWARGS = dict(num_classes=10, conv_type=3, bit_width=8, signed=False, zone=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

def extract_numeric_key(filename):
    """
    Estrae l'ID numerico unico dal nome del file.
    Esempio: da 'mul_i16_o16_1767186337_0_1a_1' estrae '1767186337'.
    """
    match = re.search(r'_(\d{8,12})_', filename)
    if match:
        return match.group(1)
    
    match_fallback = re.search(r'\b\d{8,12}\b', filename)
    return match_fallback.group(0) if match_fallback else None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
                pixel_sums = inputs.view(inputs.size(0), -1).sum(dim=1)
                print(f"[DEBUG] Batch {batch} | somma pixel per immagine: "
                      f"{[f'{v:.4f}' for v in pixel_sums.tolist()]}")
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches

def get_layer_names_from_model(model_cls, **model_kwargs):
    dummy_model = model_cls(multiplier_matrix=None, **model_kwargs)
    layer_names = []
    for m in dummy_model.modules():
        if m.__class__.__name__ == "Conv2d_custom":
            layer_names.append(m.name)
    del dummy_model
    return layer_names

def set_data_loaders(model_name: str):
    """Sets appropriate batch sizes based on the model architecture and loads data."""
    global train_loader, test_loader, _classes
    train_loader, test_loader, _classes = data_loader.get_datasets(BATCH_SIZE, model_name)

def calibration(model):
    print("Calibrating model...")
    model.eval()

    conv_modules = [m for m in model.modules() if m.__class__.__name__ == "Conv2d_custom"]
    for m in conv_modules:
        m.calibrating = True

    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= 1024 // BATCH_SIZE:
                break
            model(inputs.to(device))

    for m in conv_modules:
        m.freeze_qparams()

    model.train()  
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= 1024 // BATCH_SIZE:
                break
            model(inputs.to(device))  
    model.eval()  

def main():
    parser = argparse.ArgumentParser(description="Extract sensitivity matrix for neural networks using forward pass.")
    
    parser.add_argument("--model_name", type=str, required=True, choices=sorted(MODEL_REGISTRY.keys()),
                        help="Model architecture name to evaluate.")
    parser.add_argument("--approx_dir", type=str, required=True, 
                        help="Path to parent folder containing subfolders for each layer.")
    parser.add_argument("--quant_model_path", type=str, required=True, 
                        help="Path to the quantized model weights checkpoint (.pth file).")
    
    parser.add_argument("--full_test", action="store_true", help="If set, evaluate on the full test set.")
    parser.add_argument("--pre_training", action="store_true", help="If set, train the model with pre_training before evaluating.")
    parser.add_argument("--batch_number", type=int, default=None, help="Number of batches to evaluate for training set evaluation.")
    parser.add_argument("--csv_path", type=str, default="results.csv", help="Path to the results CSV file with metrics.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.quant_model_path):
        raise FileNotFoundError(f"Quantized model not found in {args.quant_model_path}.")

    model_cls = MODEL_REGISTRY[args.model_name]
    layer_names = get_layer_names_from_model(model_cls, **MODEL_KWARGS)
    print(f"Modello selezionato: {args.model_name}")
    print(f"Layer rilevati: {layer_names}")

    csv_metrics = {}
    if os.path.exists(args.csv_path):
        with open(args.csv_path, mode='r') as f_csv:
            reader = csv.DictReader(f_csv)
            for row in reader:
                num_key = extract_numeric_key(row['file'])
                if num_key:
                    csv_metrics[num_key] = row
    else:
        print(f"Warning: CSV file '{args.csv_path}' not found.")

    layer_candidates = [] 
    global_index_map = []
    
    for layer_idx, layer_str in enumerate(layer_names):
        l_dir = os.path.join(args.approx_dir, layer_str)
        if os.path.exists(l_dir):
            layer_num = layer_idx + 1
            
            renamed_dir = os.path.join(l_dir, "renamed_multipliers")
            os.makedirs(renamed_dir, exist_ok=True)
            
            files = sorted(glob.glob(os.path.join(l_dir, "*.npy")))
            renamed_files_for_layer = []
            
            for f in files:
                base_name = os.path.basename(f)
                name_no_ext = os.path.splitext(base_name)[0]

                temp_name_no_ext = name_no_ext
                
                suffix_to_remove = f"_{layer_str}"
                if temp_name_no_ext.endswith(suffix_to_remove):
                    temp_name_no_ext = temp_name_no_ext[:-len(suffix_to_remove)]
                
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
                
                # Match esatto sull'ID dell'istanza
                file_num_key = extract_numeric_key(base_name)
                if file_num_key and file_num_key in csv_metrics:
                    print(f"[DEBUG MATCH CSV] File '{base_name}' accoppiato con successo sulla chiave numerica: :{file_num_key}")
                    metrics = csv_metrics[file_num_key]
                    for key in ['area', 'power', 'delay', 'pda', 'mean_ae', 'mean_ae_cnn', 'max_ae', 'accuracy']:
                        if key in metrics:
                            entry[key] = float(metrics[key])
                    
                    if 'power' in entry and 'area' in entry:
                        entry['pa'] = entry['power'] * entry['area']
                else:
                    print(f"[DEBUG MATCH CSV] Nessun match trovato nel CSV per il file '{base_name}' (chiave numerica: {file_num_key})")

                global_index_map.append(entry)
            
            layer_candidates.append(renamed_files_for_layer)
        else:
            print(f"Warning: Cartella non trovata per il layer '{layer_str}' in {l_dir}")
            layer_candidates.append([])
            
    total_candidates = sum(len(cands) for cands in layer_candidates)
    if total_candidates == 0:
        raise ValueError(f"Nessun file .npy trovato sotto la cartella {args.approx_dir}")

    print(f"Inizializzazione matrice {total_candidates}x{total_candidates}...")
    S_matrix = np.zeros((total_candidates, total_candidates))
    
    def get_g_idx(layer_i, file_path):
        basename = os.path.basename(file_path)
        for idx, item in enumerate(global_index_map):
            if item["file"] == basename and item["layer_idx"] == layer_i + 1:
                return idx
        return -1

    criterion = nn.CrossEntropyLoss()

    def instantiate_and_eval(multiplier_files, baseline=False):
        start = time.time()
        quant_state_dict = torch.load(args.quant_model_path, weights_only=True)
        setup_seed(42)
        set_data_loaders(args.model_name)

        model = model_cls(
            multiplier_matrix=multiplier_files,
            **MODEL_KWARGS
        ).to(device)

        model.load_state_dict(quant_state_dict)
        calibration(model)

        if baseline:
            res = evaluate_forward_only(model, train_loader, criterion)
        else:
            res = evaluate_forward_only(model, train_loader, criterion, num_batches=args.batch_number)
        del model
        del quant_state_dict
        gc.collect()
        torch.cuda.empty_cache()
        print(f"time:{time.time() - start}")
        return res

    print("Calcolo Loss Baseline (Quantized)...")
    baseline_loss = instantiate_and_eval(None, baseline=True)
    print(f"Baseline Loss: {baseline_loss:.6f}")

    start_time = time.time()

    print("\n--- Estrazione Sensibilità Individuali ---")
    for l_idx, files in enumerate(layer_candidates):
        for f in files:
            loss = instantiate_and_eval([f])
            delta_L = loss - baseline_loss
            
            g_idx = get_g_idx(l_idx, f)
            S_matrix[g_idx, g_idx] = delta_L
            print(f"Layer {layer_names[l_idx]} (L{l_idx+1}) | File: {os.path.basename(f)} | Delta: {delta_L:.6f}")

    print("\n--- Estrazione Interazioni tra Layer ---")
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
                    
                    print(f"Coppia ({layer_names[l1_idx]}, {layer_names[l2_idx]}) | M1: {os.path.basename(f1)}, M2: {os.path.basename(f2)} | Interazione: {interazione:.6f}")

    print(f"\nEstrazione completata in {time.time() - start_time:.2f} secondi.")

    experiment_timestamp = int(time.time())
    experiment_folder_path = os.path.join(
        CURR_DIR, 
        "sensitivity_matrices", 
        f"{args.model_name}_sensitivity_n_samples_{'full_test' if args.full_test else str(BATCH_SIZE * args.batch_number if args.batch_number is not None else 1)}_{experiment_timestamp}"
    )
    os.makedirs(experiment_folder_path, exist_ok=True)

    experiment_name = f"{args.model_name}_sensitivity_matrix_{experiment_timestamp}"
    out_matrix_path = os.path.join(experiment_folder_path, f"{experiment_name}.npy")
    out_map_path = os.path.join(experiment_folder_path, f"{experiment_name}_map.json")
    
    np.save(out_matrix_path, S_matrix)
    with open(out_map_path, 'w') as map_file:
        json.dump(global_index_map, map_file, indent=4)
        
    print(f"Matrice salvata in: {out_matrix_path}")
    print(f"Mappatura indici salvata in: {out_map_path}")

if __name__ == "__main__":
    main()