import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


LOG_FILE_PATH = "log/full_cross_layer_training.log"
BASE_QUANTIZED_LOSS = 0.271785

MATRIX_DEST_PATH = "tools/multi_layer_approx/sensitivity_matrix_generation/sensitivity_matrices/full_training"

with open(LOG_FILE_PATH, 'r') as f:
    log_lines = f.readlines()


single_pattern = re.compile(r"Completato:\s+single_[a-z0-9_]+_(mae_cnn_.+?)\s+->\s+Risultato:\s+([\d\.]+)")
pair_pattern = re.compile(r"Completato:\s+[a-z0-9_]+_(mae_cnn_.+?)__x__[a-z0-9_]+_(mae_cnn_.+?)\s+->\s+Risultato:\s+([\d\.]+)")

raw_values = {}
all_labels = []

for line in log_lines:
    m_single = single_pattern.search(line)
    if m_single:
        label = m_single.group(1) + ".npy"
        val = float(m_single.group(2))
        if label not in all_labels:
            all_labels.append(label)
        raw_values[(label, label)] = val
        continue
    
    m_pair = pair_pattern.search(line)
    if m_pair:
        l1 = m_pair.group(1) + ".npy"
        l2 = m_pair.group(2) + ".npy"
        val = float(m_pair.group(3))
        raw_values[(l1, l2)] = val
        raw_values[(l2, l1)] = val


groups = defaultdict(list)
ordered_suffixes = []

for label in all_labels:
    clean_label = label.replace('.npy', '')
    suffix = "_".join(clean_label.split('_')[-2:])
    groups[suffix].append(label)
    
    if suffix not in ordered_suffixes:
        ordered_suffixes.append(suffix)

def extract_cnn_number(label):
    match = re.search(r'mae_cnn_(\d+)', label)
    if match:
        return int(match.group(1))
    return 0

ordered_labels = []

for suffix in ordered_suffixes:
    groups[suffix].sort(key=extract_cnn_number, reverse=True)
    ordered_labels.extend(groups[suffix])


n = len(ordered_labels)
matrix = np.full((n, n), np.nan)

for i, l1 in enumerate(ordered_labels):
    for j, l2 in enumerate(ordered_labels):
        if (l1, l2) in raw_values:
            matrix[i, j] = raw_values[(l1, l2)]

delta_matrix = matrix - BASE_QUANTIZED_LOSS

sens_matrix = np.full((n, n), np.nan)

for i in range(n):
    for j in range(n):
        if i == j:
            sens_matrix[i, j] = delta_matrix[i, i]
        else:
            tot = delta_matrix[i, j]
            d1 = delta_matrix[i, i]
            d2 = delta_matrix[j, j]
            
            if not np.isnan(tot) and not np.isnan(d1) and not np.isnan(d2):
                sens_matrix[i, j] = (tot - d1 - d2) / 2.0

np.nan_to_num(sens_matrix, copy=False)
np.nan_to_num(delta_matrix, copy=False)
np.nan_to_num(matrix, copy=False)

np.save(os.path.join(MATRIX_DEST_PATH, "only_loss_matrix.npy"), matrix)
np.save(os.path.join(MATRIX_DEST_PATH, "delta_loss_matrix.npy"), delta_matrix)
np.save(os.path.join(MATRIX_DEST_PATH, "sensitivity_matrix.npy"), sens_matrix)

y_labels = []
x_labels = []

for label in ordered_labels:
    clean_label = label.replace('.npy', '')
    parts = clean_label.split('_')
    
    suffix = "_".join(parts[-2:])
    layer_idx = ordered_suffixes.index(suffix) + 1
    y_labels.append(f"L{layer_idx} {clean_label}")
    
    x_labels.append(f"L{layer_idx} {clean_label}")

max_abs_val = np.nanmax(np.abs(delta_matrix))


fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    delta_matrix, 
    xticklabels=x_labels, 
    yticklabels=y_labels, 
    cmap='RdBu_r',       
    vmin=-max_abs_val, 
    vmax=max_abs_val,
    center=0,
    annot=True,          
    fmt=".3f",           
    annot_kws={"size": 7},
    cbar_kws={'label': 'Delta Loss (with respect to quantized)'},
    ax=ax
)

ax.set_title("DELTA LOSS matrix (Cross-Layer interaction)", fontsize=16, pad=15)
ax.set_ylabel("Layer & multiplier", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

current_idx = 0
for suffix in ordered_suffixes:
    current_idx += len(groups[suffix])
    if current_idx < n:
        ax.axhline(current_idx, color='black', lw=1.5)
        ax.axvline(current_idx, color='black', lw=1.5)

plt.tight_layout()
plt.savefig(os.path.join(MATRIX_DEST_PATH, "full_training_delta_loss.png"), dpi=300)
plt.show()


max_abs_val = np.nanmax(np.abs(sens_matrix))

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    sens_matrix, 
    xticklabels=x_labels, 
    yticklabels=y_labels, 
    cmap='RdBu_r',       
    vmin=-max_abs_val, 
    vmax=max_abs_val,
    center=0,
    annot=True,          
    fmt=".3f",           
    annot_kws={"size": 7},
    cbar_kws={'label': 'sensitivity (with respect to quantized)'},
    ax=ax
)

ax.set_title("INTERACTION (sensibility) matrix ", fontsize=16, pad=15)
ax.set_ylabel("Layer & multiplier", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

current_idx = 0
for suffix in ordered_suffixes:
    current_idx += len(groups[suffix])
    if current_idx < n:
        ax.axhline(current_idx, color='black', lw=1.5)
        ax.axvline(current_idx, color='black', lw=1.5)

plt.tight_layout()
plt.savefig(os.path.join(MATRIX_DEST_PATH, "full_training_interaction.png"), dpi=300)
plt.show()


es,us = np.linalg.eig(delta_matrix)
es[es<0] = 0
delta_matrix = us@np.diag(es)@us.T
delta_matrix = (delta_matrix+delta_matrix.T)/2



max_abs_val = np.nanmax(np.abs(delta_matrix))


fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    delta_matrix, 
    xticklabels=x_labels, 
    yticklabels=y_labels, 
    cmap='RdBu_r',       
    vmin=-max_abs_val, 
    vmax=max_abs_val,
    center=0,
    annot=True,          
    fmt=".3f",           
    annot_kws={"size": 7},
    cbar_kws={'label': 'Delta Loss (with respect to quantized)'},
    ax=ax
)

ax.set_title("DELTA LOSS (with PSD tranformation) matrix (Cross-Layer interaction)", fontsize=16, pad=15)
ax.set_ylabel("Layer & multiplier", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

current_idx = 0
for suffix in ordered_suffixes:
    current_idx += len(groups[suffix])
    if current_idx < n:
        ax.axhline(current_idx, color='black', lw=1.5)
        ax.axvline(current_idx, color='black', lw=1.5)

plt.tight_layout()
plt.savefig(os.path.join(MATRIX_DEST_PATH, "full_training_delta_loss_PSD.png"), dpi=300)
plt.show()

