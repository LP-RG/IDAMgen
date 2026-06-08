import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
import os
heat_maps_path = "heat_maps/npy_matrix/8_resnet_20"
def get_prob_matrix():
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

def multiplier_test(bit_width, filename,sub_x_pat_multiplier):
    scrumbled_res = np.zeros(((2**bit_width),(2**bit_width)))
    for i in range(0,(2**bit_width)):
        for y in range(0,(2**bit_width)):
            scrumbled_res[i][y] = sub_x_pat_multiplier.approx_mult(i,y)
    np.save(filename,scrumbled_res)


def get_mult_caracteristics(bit_width, filename,sub_x_pat_multiplier):
    max_error = 0
    mean_ae = 0
    mean_ae_cnn = 0
    output_exact_array = []
    max_diff = -np.inf
    prob_matrix = get_prob_matrix()
    for i in range(0,2**bit_width):
        for y in range(0,2**bit_width):
            scrumbled = sub_x_pat_multiplier.approx_mult(i,y)
            exact = i * y
            diff_no_abs = (scrumbled - exact)
            diff = abs(diff_no_abs)
            output_exact_array.append(exact)
            mean_ae += diff
            mean_ae_cnn += diff * prob_matrix[i][y]
            if(diff > max_error):
                max_error = diff
    return (mean_ae / ((2**bit_width)*(2**bit_width))), mean_ae_cnn, max_error
def execute_save(bit_width, filename):
    if 'sub_x_pat_multiplier' in sys.modules:
        sub_x_pat_multiplier = importlib.reload(sys.modules['sub_x_pat_multiplier'])
    else:
        import sub_x_pat_multiplier
    mean_ae, mean_ae_cn, max_error = get_mult_caracteristics(bit_width,filename,sub_x_pat_multiplier)
    multiplier_test(bit_width,filename,sub_x_pat_multiplier)
    return mean_ae, mean_ae_cn,max_error