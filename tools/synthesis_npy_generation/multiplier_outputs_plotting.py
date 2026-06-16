import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import argparse # Importazione aggiunta

output_plots_path = "./outputs_plot/"

def mult_output_plotting(mult_name, matrix, output_plots_path, minor_tick_step=10):
    fig, ax = plt.subplots(figsize=(6, 3)) 

        
    im = ax.imshow(matrix, cmap='Blues', origin='upper')

    # Set major ticks
    major_step_x = max(1, int(np.round(matrix.shape[1] / 5)))
    major_step_y = max(1, int(np.round(matrix.shape[0] / 5)))
    ax.set_xticks(np.arange(0, matrix.shape[1], step=major_step_x))
    ax.set_yticks(np.arange(0, matrix.shape[0], step=major_step_y))

    # Set minor ticks using MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_step))

    # Add grid lines
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='black', zorder=0)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='black', alpha=0.7, zorder=0)

    ax.set_xlabel("Weights")
    ax.set_ylabel("Activations") 
    fig.colorbar(im, ax=ax)
    
    full_save_path = os.path.join(output_plots_path, mult_name + ".png")
    plt.tight_layout() ; plt.savefig(full_save_path, bbox_inches='tight')
    plt.close(fig)

def mult_AE_plotting(mult_name, matrix, output_plots_path, minor_tick_step=10):
    fig, ax = plt.subplots(figsize=(6, 3))

    n_inputs = matrix.shape[0]
    exact_outputs = np.outer(np.arange(n_inputs), np.arange(n_inputs))
    differences = np.abs(matrix - exact_outputs)
        
    im = ax.imshow(differences, cmap='Reds', origin='upper', vmax=(np.max(differences)))

    # Set major ticks
    major_step_x = max(1, int(np.round(differences.shape[1] / 5)))
    major_step_y = max(1, int(np.round(differences.shape[0] / 5)))
    ax.set_xticks(np.arange(0, differences.shape[1], step=major_step_x))
    ax.set_yticks(np.arange(0, differences.shape[0], step=major_step_y))
    
    # Set minor ticks using MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_step))

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='black', zorder=0)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='black', alpha=0.7, zorder=0)

    ax.set_xlabel("Weights")
    ax.set_ylabel("Activations") 
    fig.colorbar(im, ax=ax) 
    
    full_save_path = os.path.join(output_plots_path, mult_name + "_AE.png")
    plt.tight_layout() ; plt.savefig(full_save_path, bbox_inches='tight')
    plt.close(fig)

def mult_binary_AE_plotting(mult_name, matrix, output_plots_path, minor_tick_step=10):
    fig, ax = plt.subplots(figsize=(6, 3))

    n_inputs = matrix.shape[0]
    exact_outputs = np.outer(np.arange(n_inputs), np.arange(n_inputs))
    differences = (np.abs(matrix - exact_outputs) >  3).astype(int)
        
    im = ax.imshow(differences, cmap='Reds', origin='upper', vmax=(np.max(differences)))

    # Set major ticks
    major_step_x = max(1, int(np.round(differences.shape[1] / 5)))
    major_step_y = max(1, int(np.round(differences.shape[0] / 5)))
    ax.set_xticks(np.arange(0, differences.shape[1], step=major_step_x))
    ax.set_yticks(np.arange(0, differences.shape[0], step=major_step_y))

    # Set minor ticks using MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_step))

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='black', zorder=0)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='black', alpha=0.7, zorder=0)

    ax.set_xlabel("Weights")
    ax.set_ylabel("Activations") 
    fig.colorbar(im, ax=ax) 
    
    full_save_path = os.path.join(output_plots_path, mult_name + "_AE_binary.png")
    plt.tight_layout()
    plt.close(fig)


def mult_RE_plotting(mult_name: str, matrix: np.ndarray, output_plots_path: str, minor_tick_step=10):
    
    fig, ax = plt.subplots(figsize=(6, 3))

    n_inputs = matrix.shape[0]
    exact_outputs = np.outer(np.arange(n_inputs), np.arange(n_inputs))

    zero_exact_mask = (exact_outputs == 0)
    
    relative_errors = np.abs(matrix - exact_outputs) * 100 / np.where(zero_exact_mask, 1, exact_outputs)

    finite_positive_errors = relative_errors[np.isfinite(relative_errors) & (relative_errors > 0)]

    if finite_positive_errors.size == 0:
        min_val = 1e-10
        max_val = 1e-9
        print(f"Warning for {mult_name}: No finite, positive relative errors found. Using default LogNorm range ({min_val}, {max_val}).")
    else:
        min_val = np.min(finite_positive_errors)
        max_val = np.max(finite_positive_errors)

        if min_val == max_val:
            if min_val > 0:
                min_val_safe = min_val * 0.99 if min_val * 0.99 < min_val else min_val - 1e-10
                max_val_safe = max_val * 1.01 if max_val * 1.01 > max_val else max_val + 1e-10
                min_val = max(1e-10, min_val_safe)
                max_val = max(min_val + 1e-10, max_val_safe)
            else:
                min_val = 1e-10
                max_val = 1e-9
            print(f"Warning for {mult_name}: All finite positive relative errors are identical. Adjusted LogNorm range to ({min_val}, {max_val}).")
        else:
            min_val = max(1e-10, min_val)

    my_norm = LogNorm(vmin=min_val, vmax=max_val)

    im = ax.imshow(relative_errors, cmap='Greens', origin='upper', norm=my_norm)
    

    # Set major ticks
    major_step_x = max(1, int(np.round(relative_errors.shape[1] / 5)))
    major_step_y = max(1, int(np.round(relative_errors.shape[0] / 5)))
    ax.set_xticks(np.arange(0, relative_errors.shape[1], step=major_step_x))
    ax.set_yticks(np.arange(0, relative_errors.shape[0], step=major_step_y))
    
    # Set minor ticks using MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_step))

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='black', zorder=0)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='black', alpha=0.7, zorder=0)
    
    ax.set_xlabel("Weights")
    ax.set_ylabel("Activations") 
    
    cbar = fig.colorbar(im, ax=ax, label='Relative Error (%) (Log Scale)') 

    if matrix.shape[0] <= 16 and matrix.shape[1] <= 16:
        for i in range(relative_errors.shape[0]):
            for j in range(relative_errors.shape[1]):
                val = relative_errors[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.0f}', 
                            ha='center', va='center', color='black')
                else:
                    ax.text(j, i, 'N/A',
                            ha='center', va='center', color='gray')

    os.makedirs(output_plots_path, exist_ok=True)
    full_save_path = os.path.join(output_plots_path, mult_name + "_RE.png")
    plt.tight_layout() ; plt.savefig(full_save_path, bbox_inches='tight')
    plt.close(fig)

def plots(mult_name,file_path, output_plots_path = output_plots_path, minor_tick_step=10):
    if(not os.path.exists(output_plots_path)):
        os.makedirs(output_plots_path)
    outputs = np.load(file_path)
    # Pass the minor_tick_step to each plotting function
    mult_AE_plotting(mult_name=mult_name, matrix=outputs, output_plots_path=output_plots_path, minor_tick_step=minor_tick_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates multiplier plots with custom tick steps."
    )
    parser.add_argument(
        "input_path",
        help="Path to the .npy file or a directory containing .npy files."
    )
    parser.add_argument(
        "--minor_tick_step",
        type=int,
        default=10,
        help="Step size for minor ticks on the plot axes (default: 10)."
    )

    args = parser.parse_args()

    input_path = args.input_path
    
    if os.path.isfile(input_path):
        plots(os.path.basename(input_path).split(".")[0], input_path, minor_tick_step=args.minor_tick_step)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith('.npy'):
                input_full_path = os.path.join(input_path, filename)
                plots(os.path.basename(filename).split(".")[0], input_full_path, minor_tick_step=args.minor_tick_step)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)