import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter

folder_path_npy = './heat_maps/npy_matrix/8_resnet_20'
folder_path_png = './heat_maps/plot/resnet_20_nw'
file_extension = '*.npy'

# ===============================
# STANDARD PLOT STYLE (stesso approccio del primo script)
# ===============================

PLOT_STYLE = {
    "figure.figsize": (10, 5),
    "figure.dpi": 1080,
    "savefig.dpi": 1080,

    "axes.titlesize": 24,
    "axes.labelsize": 24,

    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    "legend.fontsize": 14,

    "lines.linewidth": 2,

    "axes.grid": False,
}

def setup_plot_style():
    plt.rcParams.update(PLOT_STYLE)

CMAP_NAME = "viridis"

os.makedirs(folder_path_png, exist_ok=True)

def _sanitize(arr):
    arr = np.asarray(arr, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _cmap():
    cmap = plt.get_cmap(CMAP_NAME).copy()
    low = cmap(0.0)
    cmap.set_under(low)
    cmap.set_bad(low)
    return cmap

XTICKS = [0, 50, 100, 150, 200, 250]
YTICKS = [0, 50, 100, 150, 200, 250]

def _plot_matrix(mat, xlabel, ylabel, out_path, title=None, total_sum=None):

    mat = _sanitize(mat)
    cmap = _cmap()

    # Preparazione valori >0 per LogNorm
    positive_mask = mat > 0
    if not np.any(positive_mask):
        mat_safe = np.ones_like(mat) * 1e-10
    else:
        min_pos = np.min(mat[positive_mask])
        mat_safe = np.where(mat <= 0, min_pos, mat)

    vmin, vmax = np.min(mat_safe), np.max(mat_safe)
    norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)

    fig, ax = plt.subplots()

    im = ax.imshow(
        mat_safe,
        cmap=cmap,
        origin='upper',
        norm=norm,
        interpolation='nearest',
        aspect='equal'
    )

    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, pad=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)

    ax.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=14)

    # Ticks coerenti in scala log
    locator = LogLocator(numticks=6)
    cbar.locator = locator
    cbar.update_ticks()

    # Formatter della colorbar in formato scientifico (evita percentuali su log)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0e}"))
    cbar.update_ticks()

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def heat_maps_plotting():

    for file_path in glob.glob(os.path.join(folder_path_npy, file_extension)):

        h = np.load(file_path)
        h = _sanitize(h)

        mat = np.sum(h, axis=0) if h.ndim == 3 else h
        total_sum = np.sum(mat)

        out_path = re.sub(
            r"\.npy$", ".png",
            re.sub(re.escape(folder_path_npy), folder_path_png, file_path)
        )

        _plot_matrix(
            mat,
            xlabel="Weights",
            ylabel="Activations",
            out_path=out_path,
            total_sum=total_sum
        )

if __name__ == "__main__":
    setup_plot_style()
    heat_maps_plotting()