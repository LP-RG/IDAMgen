import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import re
import seaborn as sns

def visualize_sensitivity_matrix(npy_path, json_path, num_candidates=4, save_path=None, total_loss=False, psd=False):
    """
    Load and visualize sensitivity matrix using Seaborn with sorted MAE.
    """
    if not os.path.exists(npy_path) or not os.path.exists(json_path):
        print(f"Errore: File {npy_path} o {json_path} non trovati.")
        return

    matrix = np.load(npy_path)

    # --- TRASFORMAZIONE IN LOSS TOTALE (CLADO DEFAULT) ---
    if total_loss:
        print("Conversione da interazione pura a loss totale (CLADO default)...")
        diag = np.diag(matrix) # Estrae le loss individuali (L_i e L_j)

        # L_ij = 2 * S_ij + L_i + L_j
        total_loss_matrix = 2.0 * matrix + (diag[:, None] + diag[None, :])

        dim = total_loss_matrix.shape[0]
        passo = 3 # Numero di candidati/configurazioni per layer (es. 8, 7, 6 bits)

        for i in range(0, dim, passo):
            total_loss_matrix[i:i+passo, i:i+passo] = 0

        np.fill_diagonal(total_loss_matrix, diag)

        matrix = total_loss_matrix
    # -----------------------------------------------------

    # PSD transformation as in CLADO
    if psd:
        es, us = np.linalg.eig(matrix)
        es[es < 0] = 0
        matrix = us @ np.diag(es) @ us.T
        matrix = (matrix + matrix.T) / 2
    
    with open(json_path, 'r') as f:
        index_map = json.load(f)

    sorting_info = []
    for i, item in enumerate(index_map):
        layer = item.get('layer', 0)
        file_name = item.get('file', '')
        
        match = re.search(r"mae_cnn_(\d+)", file_name)

        mae_val = int(match.group(1)) if match else float('inf') 
        
        sorting_info.append((i, layer, mae_val))

    sorting_info.sort(key=lambda x: (x[1], -x[2]))
    
    new_order = [x[0] for x in sorting_info]

    index_map = [index_map[i] for i in new_order]
    

    matrix = matrix[new_order, :] # reorder rows
    matrix = matrix[:, new_order] # reorder columns

    labels = []
    for item in index_map:
        clean_name = item['file'].replace('.npy', '').strip('layer').strip('_')
        match = re.search(r"^(mul_i16_o16_\d+).*?_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)(?:_[^_]+)?$", clean_name)
        if match:
            clean_name = match.group(1) + f"_{match.group(2)}_{match.group(3)}"
        labels.append(f"L{item['layer']} {clean_name}")

    fig, ax = plt.subplots(figsize=(12, 10))

    max_abs_val = np.max(np.abs(matrix))

    sns.heatmap(
        matrix, 
        xticklabels=labels, 
        yticklabels=labels, 
        cmap='RdBu_r',       
        vmin=-max_abs_val, 
        vmax=max_abs_val,
        center=0,
        annot=True,          
        fmt=".3f",           
        annot_kws={"size": 6},
        cbar_kws={'label': 'Delta Loss (with respect to quantized)'},
        ax=ax
    )

    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    for i in range(num_candidates, len(labels), num_candidates):
        ax.axhline(i, color='black', lw=1.5)
        ax.axvline(i, color='black', lw=1.5)

    if total_loss and not psd:
        plt.title("DELTA LOSS matrix", fontsize=16, pad=20)
    if total_loss and psd:
        plt.title("DELTA LOSS matrix (with PSD transformation)", fontsize=16, pad=20)
    else:
        plt.title("INTERACTION (sensitivity) matrix", fontsize=16, pad=20)
    plt.xlabel("Layer & multiplier", fontsize=12)
    plt.ylabel("Layer & multiplier", fontsize=12)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('delta loss/interaction', rotation=270, labelpad=20)
    
    fig.tight_layout()
    
    out_img_name = "sensitivity_plot.png" 
    save_dir = save_path if save_path else os.path.dirname(npy_path)
    
    if total_loss:
        out_img_name = "mult_CLADO_delta_loss.png"
        if psd:
            out_img_name = "mult_CLADO_delta_loss_PSD.png"
    else:
        out_img_name = "mult_CLADO_interaction.png"

    out_img = os.path.join(save_dir, out_img_name)
    plt.savefig(out_img, dpi=300)
    print(f"Grafico salvato in: {out_img}")
    
    plt.show()

def analyze_best_combinations(matrix, index_map):
    """
    Analize matrix to find best PAIR of multipliers following CLADO formulation.
    """
    print("\n" + "="*60)
    print(" ANALISYS CLADO (Ranking Combinazioni)")
    print("="*60)

    risultati = []
    num_elementi = len(index_map)

    for i in range(num_elementi):
        for j in range(i + 1, num_elementi):
            layer_i = index_map[i]['layer']
            layer_j = index_map[j]['layer']

            if layer_i != layer_j:
                s_ii = matrix[i, i]
                s_jj = matrix[j, j]
                s_ij = matrix[i, j]

                # Formula CLADO: Loss(i) + Loss(j) + 2*Interazione(i,j)
                loss_totale = s_ii + s_jj + (2 * s_ij)

                nome_i = f"L{layer_i} {index_map[i]['file'].replace('.npy', '')}"
                nome_j = f"L{layer_j} {index_map[j]['file'].replace('.npy', '')}"
                nome_config = f"[{nome_i}] + [{nome_j}]"

                risultati.append((loss_totale, nome_config, s_ii, s_jj, s_ij))

    risultati.sort(key=lambda x: x[0])

    print("\nBEST 3 ESITMATED CONFIGURATION (Less loss degradation):")
    for k in range(min(3, len(risultati))):
        loss, nome, sii, sjj, sij = risultati[k]
        print(f"  {k+1}. {nome}")
        print(f"     Loss Totale:  {loss:+.5f}  (Diag1: {sii:+.4f}, Diag2: {sjj:+.4f}, Interazione: {sij:+.4f})")

    print("\nWRST COMBINATION")
    if risultati:
        loss, nome, sii, sjj, sij = risultati[-1]
        print(f"  X. {nome}")
        print(f"     Loss Totale:  {loss:+.5f}  (Diag1: {sii:+.4f}, Diag2: {sjj:+.4f}, Interazione: {sij:+.4f})\n")
    
    return risultati

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pairwise sensibility matrix")
    parser.add_argument("--matrix", type=str, default="sensitivity_matrix_resnet8.npy", help="Path to .npy matrix file")
    parser.add_argument("--map", type=str, default="sensitivity_matrix_map.json", help="Path to .json used for names")
    parser.add_argument("--cands", type=int, default=3, help="Number of multipliers tested for each layer")
    parser.add_argument("--save_path", type=str, help="directory path to save the output image")
    parser.add_argument("--analyze", action="store_true", help="Analyze best combinations based on the matrix")
    parser.add_argument("--total_loss", action="store_true", help="Use total loss instead of pure interaction for analysis")
    parser.add_argument("--data_folder", type=str, help="Path to the folder containing both json map and npy matrix files")
    parser.add_argument("--psd", action="store_true", help="Apply PSD transformation to the matrix before visualization")
    args = parser.parse_args()

    if args.data_folder:
        timestamp = args.data_folder.split("_")[-1]
        args.matrix = os.path.join(args.data_folder, f"resnet8_sensitivity_matrix_{timestamp}.npy")
        args.map = os.path.join(args.data_folder, f"resnet8_sensitivity_matrix_{timestamp}_map.json")

    visualize_sensitivity_matrix(args.matrix, args.map, args.cands, args.save_path, total_loss=args.total_loss, psd=args.psd)

    if args.analyze:
        # analyze_best_combinations(np.load(args.matrix), json.load(open(args.map, 'r')))
        pass