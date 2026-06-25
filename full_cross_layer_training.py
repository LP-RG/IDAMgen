import os
import shutil
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# config
BASE_DIR = "tools/multi_layer_approx/sensitivity_matrix_generation/experiments/21x21_CLADO_method"
OUTPUT_RUNS_DIR = "tools/multi_layer_approx/sensitivity_matrix_generation/experiments/21x21_full_training"
TRAIN_SCRIPT = "src/cnn_training.py"

MODEL_NAME = "resnet8"
CONV_TYPE = "3"
LAYER_MODE = "2"

MAX_CONCURRENT_RUNS = 4

BASE_QUANTIZED_LOSS = 0.271785


def execute_run(task):
    """Funzione worker isolata eseguita in parallelo"""
    run_name, file_paths, idx1, idx2 = task
    run_dir = os.path.join(OUTPUT_RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Copia i file necessari per questa run
    for path in file_paths:
        shutil.copy(path, run_dir)
        
    cmd = [
        "python3", TRAIN_SCRIPT,
        "--conv_type", CONV_TYPE,
        "--model_name", MODEL_NAME,
        "--multiple_layers",
        "--layer_mode", LAYER_MODE,
        "--input_path", run_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Estrazione Loss
    loss_val = np.nan
    match_loss = re.search(r"FINAL_LOSS:\s*([0-9.]+)", result.stdout)
    if match_loss:
        loss_val = float(match_loss.group(1))
    else:
        match_acc = re.search(r"FINAL_ACCURACY:\s*([0-9.]+)", result.stdout)
        if match_acc:
            loss_val = -1
            print(f"FINAL_ACCURACY found for {run_name}: {match_acc.group(1)}")
    return idx1, idx2, loss_val, run_name, result.stderr


def main():
    if not os.path.exists(BASE_DIR):
        print(f"Errore: La directory {BASE_DIR} non esiste.")
        return

    os.makedirs(OUTPUT_RUNS_DIR, exist_ok=True)

    # 1. Raccolta di tutti i file .npy
    layer_folders = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])
    all_files = [] 
    
    for layer in layer_folders:
        layer_path = os.path.join(BASE_DIR, layer)
        npy_files = sorted([f for f in os.listdir(layer_path) if f.endswith('.npy')])
        for npy in npy_files:
            all_files.append((layer, os.path.join(layer_path, npy), npy))

    def custom_sort(item):
        layer_name, _, file_name = item
        match = re.search(r"mae_cnn_(\d+)", file_name)
        mae_val = int(match.group(1)) if match else -1
        return (layer_name, -mae_val)

    all_files.sort(key=custom_sort)

    num_files = len(all_files)
    file_names = [f[2] for f in all_files]
    file_to_idx = {f[2]: i for i, f in enumerate(all_files)}
    
    loss_matrix = np.full((num_files, num_files), np.nan)

    tasks = []

    # Diagonal
    for (l1, path1, file1) in all_files:
        run_name = f"single_{l1}_{file1.replace('.npy','')}"
        idx = file_to_idx[file1]
        # task = (nome_run, lista_paths_da_copiare, indice_X, indice_Y)
        tasks.append((run_name, [path1], idx, idx))

    #  Cross-layer
    for (l1, path1, file1), (l2, path2, file2) in combinations(all_files, 2):
        if l1 != l2:
            run_name = f"{l1}_{file1.replace('.npy','')}__x__{l2}_{file2.replace('.npy','')}"
            idx1 = file_to_idx[file1]
            idx2 = file_to_idx[file2]
            tasks.append((run_name, [path1, path2], idx1, idx2))

    total_runs = len(tasks)
    print(f"Configurazione completata. Run totali in coda: {total_runs}.")
    print(f"Avvio di {MAX_CONCURRENT_RUNS} training in parallelo\n")

    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS) as executor:
        futures = {executor.submit(execute_run, task): task for task in tasks}
        
        for future in as_completed(futures):
            completed_count += 1
            idx1, idx2, loss_val, run_name, stderr = future.result()
            
            if np.isnan(loss_val):
                print(f"[{completed_count}/{total_runs}] Fallito: {run_name} (Probabile errore OOM. Check logs)")
            else:
                print(f"[{completed_count}/{total_runs}] Completato: {run_name} -> Risultato: {loss_val}")
                
            loss_matrix[idx1, idx2] = loss_val
            loss_matrix[idx2, idx1] = loss_val

    print("\nTutte le esecuzioni completate. Generazione del grafico...")
    plt.figure(figsize=(18, 16))
    
    ax = sns.heatmap(
        loss_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="RdBu_r", 
        xticklabels=file_names, 
        yticklabels=file_names,
        annot_kws={"size": 6},
        cbar_kws={'label': 'Loss'}
    )

    np.save(os.path.join(OUTPUT_RUNS_DIR, "complete_cross_layer_loss_matrix.npy"), loss_matrix)

    delta_matrix = loss_matrix - BASE_QUANTIZED_LOSS
    np.save(os.path.join(OUTPUT_RUNS_DIR, "complete_cross_layer_delta_loss_matrix.npy"), delta_matrix)

    sens_matrix = np.full((num_files, num_files), np.nan)

    for i in range(num_files):
        for j in range(num_files):
            if i == j:
                sens_matrix[i, j] = delta_matrix[i, i]
            else:
                tot = delta_matrix[i, j]
                d1 = delta_matrix[i, i]
                d2 = delta_matrix[j, j]
                
                if not np.isnan(tot) and not np.isnan(d1) and not np.isnan(d2):
                    sens_matrix[i, j] = (tot - d1 - d2) / 2.0
    
    np.save(os.path.join(OUTPUT_RUNS_DIR, "complete_cross_layer_sensitivity_matrix.npy"), sens_matrix)
    
    plt.title('Complete Cross-Layer Sensitivity Matrix (Parallel Run)')
    plt.xlabel('Layer i Configuration')
    plt.ylabel('Layer j Configuration')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_image = "complete_cross_layer_sensitivity_matrix.png"
    plt.savefig(output_image, dpi=300)
    print(f"Matrice completata e salvata come {output_image}")

if __name__ == "__main__":
    main()