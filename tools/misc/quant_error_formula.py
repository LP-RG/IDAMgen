import numpy as np
import matplotlib.pyplot as plt
import os

heat_maps_path = "heat_maps/npy_matrix/8bit_resnet_8"

layer_to_heatmap_file_resnet8 = {
    "conv1": "s_8.npy",
    "layer1.conv1": "1_1.npy",
    "layer1.conv2": "1_2.npy",
    "layer2.conv1": "2_1.npy",
    "layer2.conv2": "2_2.npy",
    "layer2.shortcut.0": "2_s.npy",
    "layer3.conv1": "3_1.npy",
    "layer3.conv2": "3_2.npy",
    "layer3.shortcut.0": "3_s.npy"
}

def get_prob_matrix(layer_name=None):

    if layer_name is not None:
        heat_map_file = layer_to_heatmap_file_resnet8.get(layer_name)
        if heat_map_file is None:
            raise ValueError(f"No heatmap file found for layer: {layer_name}")
        matrix = np.load(os.path.join(heat_maps_path, heat_map_file))
        matrix = matrix.sum(axis=0)
        prob_matrix = matrix / np.sum(matrix)
        return prob_matrix
    
    heat_map_files = sorted([f for f in os.listdir(heat_maps_path) if f.endswith(".npy")])
    cumulative_heatmap = None
    for f in heat_map_files:
        matrix = np.load(os.path.join(heat_maps_path, f))
        matrix =matrix.sum(axis = 0)
        if cumulative_heatmap is None:
            cumulative_heatmap = matrix.copy()
        else:
            cumulative_heatmap += matrix
    prob_matrix = cumulative_heatmap / np.sum(cumulative_heatmap)
    return prob_matrix

def generate_and_save_error_matrix(target_bit, base_bit=8, exact=False, layer_name=None):
    if target_bit >= base_bit or target_bit < 1:
        raise ValueError("target_bit must be between 1 and 7 (inclusive) for INT{base_bit} to INT{target_bit} conversion.")
        
    mode_str = "exact" if exact else "approx"

    filename = f"error_matrix_int{target_bit}_{mode_str}.png"



    vals_base = np.arange(2 ** base_bit, dtype=np.int32)
    X_base, W_base = np.meshgrid(vals_base, vals_base, indexing='ij')

    zp_i_base = 0
    zp_w_base = 0


    exact_mac = (X_base - zp_i_base) * (W_base - zp_w_base)
    
    if exact:
        val_base = (2 ** base_bit) - 1
        val_small = (2 ** target_bit) - 1
        
        # Round((X * (2^b_small - 1)) / (2^b_base - 1))
        x_small_rounded = np.round(((X_base - zp_i_base) * val_small) / val_base)
        # Round((W * (2^b_small - 1)) / (2^b_base - 1))
        w_small_rounded = np.round(((W_base - zp_w_base) * val_small) / val_base)
        
        # ((2^b_base - 1) / (2^b_small - 1))^2
        scale_ratio_sq = (val_base / val_small) ** 2
        approx_mac = (x_small_rounded * w_small_rounded) * scale_ratio_sq
        
    else:
        shift = base_bit - target_bit
        total_left_shift = shift * 2
        approx_mac = (((X_base - zp_i_base) >> shift) * ((W_base - zp_w_base) >> shift)) << total_left_shift
    
    error_matrix = np.abs(exact_mac - approx_mac)
    

    prob_matrix = get_prob_matrix(layer_name=layer_name)
    mean_ae = np.sum(error_matrix * prob_matrix)

    
    mode_title = "exact" if exact else "approx"
    print(f"--- Results for INT{target_bit} ({mode_title}) ---")
    print(f"Quantization max_ae on ResNet  = {np.max(error_matrix):.4f}")
    if mean_ae is not None:
        print(f"Quantization mean_ae on ResNet with {target_bit} bits = {mean_ae:.4f}\n")
    else:
        print("Quantization mean_ae: not calculable (heatmap files not found)\n")

    plt.figure(figsize=(8, 6))
    plt.imshow(error_matrix, cmap='inferno') 
    
    cbar = plt.colorbar()
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)
    
    plt.title(f'Error Matrix of INT{target_bit} Multiplier {mode_title}')
    plt.xlabel(f'Weight (W{base_bit})')
    plt.ylabel(f'Activation (X{base_bit})')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Error matrix saved successfully as: {filename}\n")

if __name__ == "__main__":

    quantization_params_resnet_20 = {
        "conv1": {
            "activation": {"scale": 0.007843, "zp_neg": -127.0},
            "weight": {"scale": 0.012172, "zp_neg": -131.0}
        },
        "layer1.0.conv1": {
            "activation": {"scale": 0.026259, "zp_neg": 0.0},
            "weight": {"scale": 0.007700, "zp_neg": -132.0}
        },
        "layer1.0.conv2": {
            "activation": {"scale": 0.024556, "zp_neg": 0.0},
            "weight": {"scale": 0.007342, "zp_neg": -116.0}
        },
        "layer1.1.conv1": {
            "activation": {"scale": 0.042350, "zp_neg": 0.0},
            "weight": {"scale": 0.006000, "zp_neg": -143.0}
        },
        "layer1.1.conv2": {
            "activation": {"scale": 0.026059, "zp_neg": 0.0},
            "weight": {"scale": 0.006070, "zp_neg": -138.0}
        },
        "layer1.2.conv1": {
            "activation": {"scale": 0.047632, "zp_neg": 0.0},
            "weight": {"scale": 0.008142, "zp_neg": -142.0}
        },
        "layer1.2.conv2": {
            "activation": {"scale": 0.022505, "zp_neg": 0.0},
            "weight": {"scale": 0.004565, "zp_neg": -119.0}
        },
        "layer2.0.conv1": {
            "activation": {"scale": 0.052190, "zp_neg": 0.0},
            "weight": {"scale": 0.004963, "zp_neg": -132.0}
        },
        "layer2.0.conv2": {
            "activation": {"scale": 0.019538, "zp_neg": 0.0},
            "weight": {"scale": 0.005943, "zp_neg": -110.0}
        },
        "layer2.0.shortcut.0": {
            "activation": {"scale": 0.052190, "zp_neg": 0.0},
            "weight": {"scale": 0.009377, "zp_neg": -91.0}
        },
        "layer2.1.conv1": {
            "activation": {"scale": 0.038450, "zp_neg": 0.0},
            "weight": {"scale": 0.004745, "zp_neg": -108.0}
        },
        "layer2.1.conv2": {
            "activation": {"scale": 0.016121, "zp_neg": 0.0},
            "weight": {"scale": 0.003366, "zp_neg": -116.0}
        },
        "layer2.2.conv1": {
            "activation": {"scale": 0.041981, "zp_neg": 0.0},
            "weight": {"scale": 0.003766, "zp_neg": -132.0}
        },
        "layer2.2.conv2": {
            "activation": {"scale": 0.016343, "zp_neg": 0.0},
            "weight": {"scale": 0.003722, "zp_neg": -115.0}
        },
        "layer3.0.conv1": {
            "activation": {"scale": 0.045597, "zp_neg": 0.0},
            "weight": {"scale": 0.003944, "zp_neg": -127.0}
        },
        "layer3.0.conv2": {
            "activation": {"scale": 0.017885, "zp_neg": 0.0},
            "weight": {"scale": 0.003844, "zp_neg": -120.0}
        },
        "layer3.0.shortcut.0": {
            "activation": {"scale": 0.045597, "zp_neg": 0.0},
            "weight": {"scale": 0.005138, "zp_neg": -119.0}
        },
        "layer3.1.conv1": {
            "activation": {"scale": 0.033538, "zp_neg": 0.0},
            "weight": {"scale": 0.003571, "zp_neg": -125.0}
        },
        "layer3.1.conv2": {
            "activation": {"scale": 0.019629, "zp_neg": 0.0},
            "weight": {"scale": 0.003277, "zp_neg": -119.0}
        },
        "layer3.2.conv1": {
            "activation": {"scale": 0.051688, "zp_neg": 0.0},
            "weight": {"scale": 0.002601, "zp_neg": -120.0}
        },
        "layer3.2.conv2": {
            "activation": {"scale": 0.015851, "zp_neg": 0.0},
            "weight": {"scale": 0.002120, "zp_neg": -113.0}
        }
    }
    quantization_params_resnet_8 = {
        "conv1": {
            "activation": {"scale": 0.007843, "zp_neg": -127.0},
            "weight": {"scale": 0.017024, "zp_neg": -131.0}
        },
        "layer1.conv1": {
            "activation": {"scale": 0.051017, "zp_neg": 0.0},
            "weight": {"scale": 0.013335, "zp_neg": -142.0}
        },
        "layer1.conv2": {
            "activation": {"scale": 0.033972, "zp_neg": 0.0},
            "weight": {"scale": 0.008256, "zp_neg": -119.0}
        },
        "layer2.conv1": {
            "activation": {"scale": 0.051547, "zp_neg": 0.0},
            "weight": {"scale": 0.006608, "zp_neg": -147.0}
        },
        "layer2.conv2": {
            "activation": {"scale": 0.028924, "zp_neg": 0.0},
            "weight": {"scale": 0.005880, "zp_neg": -130.0}
        },
        "layer2.shortcut.0": {
            "activation": {"scale": 0.051547, "zp_neg": 0.0},
            "weight": {"scale": 0.008176, "zp_neg": -115.0}
        },
        "layer3.conv1": {
            "activation": {"scale": 0.032074, "zp_neg": 0.0},
            "weight": {"scale": 0.004661, "zp_neg": -130.0}
        },
        "layer3.conv2": {
            "activation": {"scale": 0.019003, "zp_neg": 0.0},
            "weight": {"scale": 0.004198, "zp_neg": -126.0}
        },
        "layer3.shortcut.0": {
            "activation": {"scale": 0.032074, "zp_neg": 0.0},
            "weight": {"scale": 0.003511, "zp_neg": -110.0}
        }
    }

    # for layer_name, _ in quantization_params_resnet_8.items():
    #     generate_and_save_error_matrix(target_bit=7, exact=True, layer_name=layer_name)

    for tb in range(1, 8):
        generate_and_save_error_matrix(target_bit=tb, exact=True)

    #generate_and_save_error_matrix(target_bit=7, exact=True)
    