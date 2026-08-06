import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import math
import argparse
import ast

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAT_MAPS_PATH_RESNET_20 = os.path.join(ROOT_DIR, "heat_maps/npy_matrix/8bit_8_resnet_20")
HEAT_MAPS_PATH_RESNET_8 = os.path.join(ROOT_DIR, "heat_maps/npy_matrix/8bit_8_resnet_8")
HEAT_MAPS_PATH_ALEXNET = os.path.join(ROOT_DIR, "heat_maps/npy_matrix/8bit_alex_net")
HEAT_MAPS_PATH_BOUND = os.path.join(ROOT_DIR, "heat_maps/npy_matrix/8bit_alex_net")

MAX_POSSIBLE_ERROR = 255*255

def get_layer_prob_matrix(heat_map_file):
    matrix = np.load(heat_map_file)
    total = np.sum(matrix)
    if total == 0:
        return np.zeros_like(matrix.sum(axis=0))
    return matrix.sum(axis=0) / total

def get_prob_matrix(heat_map_path):
    heat_map_files = sorted([f for f in os.listdir(heat_map_path) if f.endswith(".npy")])
    cumulative_heatmap = None
    for f in heat_map_files:
        matrix = np.load(os.path.join(heat_map_path, f))
        matrix = matrix.sum(axis=0)
        if cumulative_heatmap is None:
            cumulative_heatmap = matrix.copy()
        else:
            cumulative_heatmap += matrix

    total = np.sum(cumulative_heatmap)
    if total == 0:
        return np.zeros_like(cumulative_heatmap)
    return cumulative_heatmap / total

def plot_aet_assingment(matrix, name_file, block_size, save_plot=False):
    matrix = np.array(matrix)
    if matrix.ndim == 1:
        side = int(np.sqrt(matrix.size))
        if side * side != matrix.size:
            raise ValueError("Input array length must be a perfect square to plot AET assignment.")
        matrix = matrix.reshape(side, side)
        if block_size is None:
            block_size = 256 // side

    # ---- espansione a 256×256 ----
    expanded_matrix = np.kron(matrix, np.ones((block_size, block_size)))

    plt.figure(figsize=(10,10))  # dimensione stabile e leggibile

    plt.imshow(expanded_matrix, cmap='Reds')
    plt.xlabel("weights")
    plt.ylabel("activations")

    # ---- testo al centro dei blocchi ----
    h, w = matrix.shape

    for i in range(h):
        for j in range(w):

            value = matrix[i, j]

            center_i = i * block_size + (block_size - 1) / 2
            center_j = j * block_size + (block_size - 1) / 2

            plt.text(center_j, center_i,
                     f"{value:.0f}",
                     ha='center', va='center',
                     color='black', fontsize=8, fontweight='bold')

    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    #print(name_file)
    if save_plot:
        plt.savefig(name_file, dpi=300)
    plt.show()
    plt.close()
    


def upper_bound_calculation(matrix, block_size):
    matrix = np.array(matrix)

    # ---- espansione a 256×256 ----
    expanded_matrix = np.kron(matrix, np.ones((block_size, block_size))) 
    probability_matrix = get_prob_matrix(HEAT_MAPS_PATH_BOUND)
    total_sum_u = 0
    total_sum_cnn = 0
    for x in range(expanded_matrix.shape[0]):
        for y in range(expanded_matrix.shape[1]):
            ub = min(expanded_matrix[x][y], x*y)
            total_sum_u += ub
            total_sum_cnn += ub * probability_matrix[x][y]

    print(f"meanAEu = {total_sum_u/(expanded_matrix.shape[0])**2}")
    print(f"meanAEcnn = {total_sum_cnn}")
    print(f"meanAEratio = {(total_sum_u/(expanded_matrix.shape[0])**2)/ total_sum_cnn}")




def gen_array_sqrt_variable(square_size,AET,alpha, prob_matrix=None, heatmap_path=HEAT_MAPS_PATH_ALEXNET, return_matrix=False):
    if prob_matrix is None:
        prob_matrix = get_prob_matrix(heatmap_path)

    number_of_zones = (2**16) / (square_size)**2
    blocks = prob_matrix.reshape(2**8//square_size, square_size, 2**8//square_size, square_size)
    agg = blocks.sum(axis=(1, 3))
    real_AET = (AET * number_of_zones) / np.sum(np.power(agg,1-alpha))
    sum_for_debug = 0
    AET_array = np.zeros(agg.shape)
    for i in range(0,agg.shape[0]):
        for j in range(0,agg.shape[1]):
            if agg[i][j] == 0:
                AET_array[i][j] = MAX_POSSIBLE_ERROR
            else:
                AET_array[i][j] = real_AET / (number_of_zones * (agg[i][j])**alpha)

                if AET_array[i][j] > MAX_POSSIBLE_ERROR:
                    AET_array[i][j] = MAX_POSSIBLE_ERROR

                sum_for_debug += 1 / (agg[i][j])**alpha
    if return_matrix:
        return AET_array
    AET_array_flatten = AET_array.astype(int)
    AET_array_flatten = AET_array_flatten.flatten().tolist()
    return(AET_array_flatten)

def check_error(metric_type, threshold, layer_1, layer_2) -> bool:

    layer_1_values = layer_1['values']
    layer_2_values = layer_2['values']

    layer_1_keys = layer_1['layers_keys']
    layer_2_keys = layer_2['layers_keys']

    def print_offending_cells(errors, threshold, error_type):
        offending_cells = np.where(errors > threshold)[0]
        print(f"First 10 cells out of {len(offending_cells)} with {error_type} error above the threshold between {layer_1_keys[0][3]} and {layer_2_keys[0][3]}:")
        distance = np.mean(errors)
        print(f"  Mean {error_type} error: {distance:.4f}")
        for i in offending_cells[:10]:
            print(f"  Cell {i}: {errors[i]}")

    if metric_type == "relative":

        distance = 0

        max_vals = np.maximum(layer_1_values, layer_2_values)
        max_vals[max_vals == 0] = 1e-9 
        relative_errors = np.abs(layer_1_values - layer_2_values) / max_vals

        #if np.sum(relative_errors <= threshold) >= 0.8 * len(relative_errors):
        if np.all(relative_errors <= threshold):
            return True
        else:
            print_offending_cells(relative_errors, threshold, "relative")

    if metric_type == "absolute":
        absolute_errors = np.abs(layer_1_values - layer_2_values)
        #if np.sum(absolute_errors <= threshold) >= 0.8 * len(absolute_errors):
        if np.all(absolute_errors <= threshold):
            return True
        else:
            print_offending_cells(absolute_errors, threshold, "absolute")

    return False

def analyze_merge_layer_similarity(values_map: dict, eps_rel: float=0.20, metric_type: str="relative") -> list:
    items = list(values_map.items())
    num_layers = len(items)
    

    clusters = []
    for key, values in items:
        clusters.append({
            'layers_keys': [key],
            'values': np.array(values, dtype=float)
        })
    
    epsilon = 1e-9 
    
    i = 0
    while i < len(clusters):
        j = i + 1
        while j < len(clusters):
            mat_A = clusters[i]['values']
            mat_B = clusters[j]['values']
            
            
            if check_error(metric_type, eps_rel, clusters[i], clusters[j]):
                clusters[i]['layers_keys'].extend(clusters[j]['layers_keys'])
                
                clusters[i]['values'] = np.minimum(mat_A, mat_B)
                
                clusters.pop(j)
                
                i = 0 
                break
            else:
                j += 1
        else:
            i += 1
    if metric_type == "relative":
        print(f"\n--- ANALISI MERGE (metrica: {metric_type}, Tolleranza: {eps_rel*100}%) ---")
    else:
        print(f"\n--- ANALISI MERGE (metrica: {metric_type}, Tolleranza: {eps_rel}) ---")
    
    print(f"Layer originali: {num_layers} -> Cluster finali: {len(clusters)}\n")
    
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1}:")
        print(f"  Layer inclusi ({len(cluster['layers_keys'])}):")
        for key in cluster['layers_keys']:
            print(f"    - {key[3] if isinstance(key, tuple) else key}")
        print("-" * 40)
        
    return clusters

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate AET arrays for zone based partitioning.')
    parser.add_argument('--input-array', type=str, default=None, help='Flat AET array to plot directly, e.g. "[1,2,3,...]"')
    parser.add_argument('--pwrs', nargs='+', type=float, default=[0.3,0.5], help='List of power levels to generate AET arrays for.')
    parser.add_argument('--aets', nargs='+', type=int, default=[450, 650, 850], help='List of AET values to generate AET arrays for.')
    parser.add_argument('--edges', nargs='+', type=int, default=[16], help='List of edge sizes to generate AET arrays for.')
    parser.add_argument('--heatmap_path', type=str, default=HEAT_MAPS_PATH_RESNET_8, help='Path to the heatmap .npy files.')
    parser.add_argument('--per-layer', action='store_true', help='If set, process each heatmap file in the path separately (call get_layer_prob_matrix per file)')
    
    parser.add_argument('--plot', action='store_true', help='If set, show plot.') 
    
    parser.add_argument('--analyze-merge', action='store_true', help='If set, analyze and merge similar layers based on their AET arrays.')
    parser.add_argument('--et', type=float, default=0.20, help='Error threshold for merging layers (used if --analyze_merge is set).')
    parser.add_argument('--metric-type', type=str, choices=['relative', 'absolute'], default='relative', help='Type of error metric to use for merging layers (used if --analyze_merge is set).')

    args = parser.parse_args()

    if args.input_array is not None:
        input_array = np.array(ast.literal_eval(args.input_array))
        plot_aet_assingment(input_array, "input_array_plot.png", None, save_plot=True)
        print("Saved plot to input_array_plot.png")
        raise SystemExit(0)

    heatmap_files = sorted([f for f in os.listdir(args.heatmap_path) if f.endswith('.npy')])

    values_map = {}

    if args.per_layer:
        for f in heatmap_files:
            prob = get_layer_prob_matrix(os.path.join(args.heatmap_path, f))
            for pwr in args.pwrs:
                for aet in args.aets:
                    for edge in args.edges:
                        if args.plot:
                            aet_mat = gen_array_sqrt_variable(edge, aet, pwr, prob_matrix=prob, return_matrix=True)
                            plot_aet_assingment(aet_mat, f"{f}_edge_{edge}.png", edge, save_plot=True)
                            values = aet_mat.astype(int).flatten().tolist()
                        else:
                            values = gen_array_sqrt_variable(edge, aet, pwr, prob_matrix=prob)
                        values_map[(pwr, aet, edge, f)] = values                        
                        print(",\n")
                        print("{\n")
                        print(f'"descr":"theoretical_pwr_{pwr}_AET_{int(aet)}_file_{f}_zone_{int((256/edge)**2)}",\n')
                        print(f'"values":{values}\n')
                        print("}\n")
    else:
        for pwr in args.pwrs:
            for aet in args.aets:
                for edge in args.edges:
                    prob = get_prob_matrix(args.heatmap_path)
                    if args.plot:
                        aet_mat = gen_array_sqrt_variable(edge, aet, pwr, prob_matrix=prob, return_matrix=True)
                        plot_aet_assingment(aet_mat, f"aggregate_edge_{edge}.png", edge, save_plot=True)
                        values = aet_mat.astype(int).flatten().tolist()
                    else:
                        values = gen_array_sqrt_variable(edge, aet, pwr, prob_matrix=prob)
                    print(",\n")
                    print("{\n")
                    print(f'"descr":"theoretical_pwr_{pwr}_AET_{int(aet)}_zone_{int((256/edge)**2)}",\n')
                    print(f'"values":{values}\n')
                    print("}\n")
    
    if args.analyze_merge:
        analyze_merge_layer_similarity(values_map, eps_rel=args.et, metric_type=args.metric_type)
    
    
    #matrix = gen_array_sqrt_variable(32, AET, pwr)
    """plot_aet_assingment(matrix, f"alexnet_aet_{AET}_pwr_{pwr}.png", 32)
    upper_bound_calculation(matrix, 32)
    print("#############")
    """
