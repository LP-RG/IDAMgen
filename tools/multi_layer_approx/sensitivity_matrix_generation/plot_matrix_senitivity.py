import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import re

def visualize_sensitivity_matrix(npy_path, json_path, num_candidates=4, save_path=None):
    """
    Load and visualize sensitivity matrix.
    """
    if not os.path.exists(npy_path) or not os.path.exists(json_path):
        print(f"Errore: File {npy_path} o {json_path} non trovati.")
        return

    matrix = np.load(npy_path)
    
    with open(json_path, 'r') as f:
        index_map = json.load(f)

    labels = []
    for item in index_map:
        clean_name = item['file'].replace('.npy', '').strip('layer').strip('_')
        match = re.search(r"^(mul_i16_o16_\d+).*?_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)(?:_[^_]+)?$", clean_name)
        if match:
            clean_name = match.group(1) + f"_{match.group(2)}_{match.group(3)}"
        labels.append(f"L{item['layer']} {clean_name}")

    fig, ax = plt.subplots(figsize=(12, 10))

    max_abs_val = np.max(np.abs(matrix))

    cax = ax.imshow(matrix, cmap="plasma", vmin=-max_abs_val, vmax=max_abs_val)

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Delta Loss (whith respect to quantized)', rotation=270, labelpad=20)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if abs(val) > (max_abs_val * 0.6) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    for i in range(num_candidates, len(labels), num_candidates):
        ax.axhline(i - 0.5, color='black', lw=2)
        ax.axvline(i - 0.5, color='black', lw=2)

    plt.title("Hardware sensibility matrix (Cross-Layer interaction)", fontsize=16, pad=20)
    plt.xlabel("Layer & multiplier", fontsize=12)
    plt.ylabel("Layer & multiplier", fontsize=12)
    
    fig.tight_layout()
    
    out_img_name = "sensitivity_plot.png" 
    save_dir = save_path if save_path else os.path.dirname(npy_path)
    out_img = os.path.join(save_dir, out_img_name)
    plt.savefig(out_img, dpi=300)
    print(f"Grafico salvato in alta risoluzione: {out_img}")
    
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
    parser.add_argument("--cands", type=int, default=4, help="Number of multipliers tested for each layer")
    parser.add_argument("--save_path", type=str, help="directory path to save the output image")
    args = parser.parse_args()

    visualize_sensitivity_matrix(args.matrix, args.map, args.cands, args.save_path)

    analyze_best_combinations(np.load(args.matrix), json.load(open(args.map, 'r')))