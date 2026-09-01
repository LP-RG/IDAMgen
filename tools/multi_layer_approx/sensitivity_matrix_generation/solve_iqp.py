import argparse
import json
import numpy as np
import cvxpy as cp
import gurobipy as gp
import os
import glob
import shutil
import sys
import csv
import torch  
import mat_mul
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

# Total multiplications for each layer: {layer_name: (mults_per_op, total_mults_in_layer)}
LAYER_MULTS = {}

# EXACT_DATA global dictionary
EXACT_DATA = {
    "area": 358.568,
    "power": 0.0005713708815165,
    "delay": 0.91213,
    "pa": 0.20487531424360662,
    "pda": 0.1868729203810225,
    "mean_ae": 0.0,
    "mean_ae_cnn": 0.0,
    "max_ae": 0.0,
    "accuracy": 87.39
}
with open("gurobi.lic", "r", encoding="utf-8") as file:
    license = dict(line.strip().split("=", 1) for line in file if "=" in line)


GUROBI_WLS_OPTIONS = {
    "WLSACCESSID": license.get("WLSACCESSID"),
    "WLSSECRET": license.get("WLSSECRET"),
    "LICENSEID": license.get("LICENSEID"),
}

def get_layers_stats(model_cls, **model_kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = model_cls(multiplier_matrix=None, **model_kwargs).to(device)
    dummy_model.eval()
    
    def hook_fn(module, input, output):
        mults_per_op = module.kernel_size[0] * module.kernel_size[1] * module.channel_in
        out_h, out_w = output.shape[2], output.shape[3]
        total_mults = mults_per_op * module.channel_out * out_h * out_w
        LAYER_MULTS[module.name] = (mults_per_op, total_mults)

    hooks = []
    for m in dummy_model.modules():
        if m.__class__.__name__ == "Conv2d_custom":
            hooks.append(m.register_forward_hook(hook_fn))

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    
    with torch.no_grad():
        dummy_model(dummy_input)

    for h in hooks:
        h.remove()
        
    del dummy_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_config_metrics(selected_indices, mapping, active_layers, N=100000):
    total_moltiplicatori = sum(LAYER_MULTS[l][0] for l in active_layers)
    total_moltiplicazioni = sum(LAYER_MULTS[l][1] for l in active_layers)
    print(f"total moltiplicatori {total_moltiplicatori} , total moltiplicazioni {total_moltiplicazioni}")
    area = 0.0
    power = 0.0
    delay_case1 = 0.0
    layer_delays = []

    for idx in selected_indices:
        layer_str = mapping[idx]['layer']
        n_moltiplicatori_layer = LAYER_MULTS[layer_str][0]
        n_mults_layer = LAYER_MULTS[layer_str][1]
        
        area += mapping[idx]['area'] * n_moltiplicatori_layer
        power += mapping[idx]['power'] * n_moltiplicatori_layer

        # Tempo di esecuzione per blocco del singolo layer
        current_layer_delay = (n_mults_layer / n_moltiplicatori_layer) * mapping[idx]['delay']
        
        delay_case1 += current_layer_delay
        layer_delays.append(current_layer_delay)

    area_config = area / total_moltiplicatori
    power_config = power / total_moltiplicatori
    
    bottleneck_delay = max(layer_delays) if layer_delays else 0.0
    
    if N > 0:
        delay_case2 = delay_case1 + (N - 1) * bottleneck_delay
    else:
        delay_case2 = bottleneck_delay

    return area_config, power_config, delay_case1, delay_case2

def save_solution_csv(out_dir, config_id, selected_indices, mapping, active_layers):
    csv_path = os.path.join(out_dir, "solution.csv")

    area_config, power_config, delay_case1, delay_case2 = compute_config_metrics(
        selected_indices, mapping, active_layers
    )

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "multiplier_file", "area", "power", "delay", "pda", "delay_stream_case2"])

        for idx in selected_indices:
            item = mapping[idx]
            writer.writerow([
                item['layer'],
                item['file'],
                item['area'],
                item['power'],
                item['delay'],
                item['pda'],
                "",
            ])

        # Riga di riepilogo configurazione
        writer.writerow([
            config_id,
            "",
            area_config,
            power_config,
            delay_case1,
            "",
            delay_case2,
        ])

    print(f"Salvato riepilogo configurazione in: {csv_path}")

def load_weighted_costs(mapping, metric="area"):
    if metric not in EXACT_DATA:
        raise KeyError(f"Metric '{metric}' not found in EXACT_DATA. Please add it to the global dictionary.")
    
    exact_unit_val = EXACT_DATA[metric]

    weighted_costs = np.zeros(len(mapping))
    baseline_cost = 0.0
    
    active_layers = set([m['layer'] for m in mapping])
    
    for layer_str in active_layers:
        baseline_cost += exact_unit_val * LAYER_MULTS[layer_str][0]
    for i, item in enumerate(mapping):
        layer_str = item['layer']
        
        unit_val = item.get(metric)
        if unit_val is None:
            raise ValueError(f"Metric '{metric}' not found for file: {item['file']}")
            
        weighted_costs[i] = unit_val * LAYER_MULTS[layer_str][0]
    return weighted_costs, baseline_cost

def make_matrix_psd(G, eps=1e-6):
    G_sym = (G + G.T) / 2
    es, us = np.linalg.eigh(G_sym)
    es = np.clip(es, eps, None) 
    G_psd = us @ np.diag(es) @ us.T
    return (G_psd + G_psd.T) / 2

def solve_iqp(G, mapping, weighted_costs, mode, target_val, solver='GUROBI', metric="area"):
    """Sets up and solves the IQP problem using CVXPY."""
    
    n_vars = len(mapping)
    alpha = cp.Variable(n_vars, boolean=True)
    
    G_psd = make_matrix_psd(G)
    G_wrapped = cp.atoms.affine.wraps.psd_wrap(G_psd)
    
    loss_expr = cp.quad_form(alpha, G_wrapped)
    cost_expr = alpha @ weighted_costs
    
    constraints = []
    
    layers = sorted(list(set([m['layer_idx'] for m in mapping])))
    for l in layers:
        idx = [i for i, m in enumerate(mapping) if m['layer_idx'] == l]
        constraints.append(cp.sum(alpha[idx]) == 1)

    if mode == 'min_loss':
        objective = cp.Minimize(loss_expr)
        constraints.append(cost_expr <= target_val)
    elif mode == 'min_area':
        objective = cp.Minimize(cost_expr)
        constraints.append(loss_expr <= target_val)
    else:
        raise ValueError("Mode not supported. Use 'min_loss' or 'min_area'.")

    prob = cp.Problem(objective, constraints)

    with gp.Env(params=GUROBI_WLS_OPTIONS) as env:
        try:
            prob.solve(solver=cp.GUROBI, verbose=True, TimeLimit=6000, NonConvex=2, env=env)
        except Exception as e:
            print(f"Solver error {solver}: {e}. Falling back to default.")
            prob.solve(verbose=True)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solution status: {prob.status}. Unable to find a valid configuration.")
        return None

    selected_indices = np.where(alpha.value > 0.5)[0]
    
    print("\n--- Optimization Results ---")
    print(f"Status: {prob.status}")
    print(f"Estimated Loss (Delta): {loss_expr.value:.6f}")
    print(f"Total Weighted Cost ({metric}): {cost_expr.value:.4e}")
    
    print("\nSelected Configuration:")
    for idx in selected_indices:
        l_idx = mapping[idx]['layer_idx']
        layer_str = mapping[idx]['layer']
        print(f"Layer {l_idx} ({layer_str}) -> {mapping[idx]['file']} | Weighted Cost ({metric}): {weighted_costs[idx]:.4e}")

    return alpha.value

def main():
    parser = argparse.ArgumentParser(description="IQP Optimization for Approximate Multipliers with dynamic metrics.")
    parser.add_argument("--model_name", type=str, required=True, choices=sorted(MODEL_REGISTRY.keys()),
                        help="Model architecture name to evaluate.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment folder containing the _matrix.npy and _map.json")
    parser.add_argument("--metric", type=str, default="area", help="Hardware metric to optimize (e.g., area, power, delay, pa, pda)")
    
    parser.add_argument("--mode", type=str, default='min_loss', choices=['min_loss', 'min_area'])
    parser.add_argument("--target_perc", type=float, required=True, help="Target percentage (e.g., 0.8 for 80% of baseline budget in min_loss, or absolute threshold in min_area)")
    parser.add_argument("--solver", type=str, default="GUROBI", help="Solver to use")

    args = parser.parse_args()
    model_cls = MODEL_REGISTRY[args.model_name]
    get_layers_stats(model_cls, **MODEL_KWARGS)

    npy_files = glob.glob(os.path.join(args.experiment_dir, "*matrix*.npy"))
    json_files = glob.glob(os.path.join(args.experiment_dir, "*map.json"))
    
    if not npy_files or not json_files:
        raise FileNotFoundError(f"Could not find the matrix .npy or map .json inside '{args.experiment_dir}'. Ensure the extraction script finished successfully.")
    
    matrix_path = npy_files[0]
    map_path = json_files[0]

    G_matrix = np.load(matrix_path)
    with open(map_path, 'r') as f:
        mapping = json.load(f)

    weighted_costs, baseline_cost = load_weighted_costs(mapping, metric=args.metric)

    if args.mode == 'min_loss':
        target_val = baseline_cost * args.target_perc
        print(f"Optimizing metric: {args.metric}")
        print(f"Calculated baseline {args.metric}: {baseline_cost:.4e}")
        print(f"Cost target ({args.target_perc*100}%): {target_val:.4e}")
    else:
        target_val = args.target_perc
        print(f"Optimizing metric: {args.metric}")
        print(f"Cost minimization. Loss constraint <= {target_val}")

    alpha_result = solve_iqp(G_matrix, mapping, weighted_costs, args.mode, target_val, solver=args.solver, metric=args.metric)

    if alpha_result is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        exp_folder_name = os.path.basename(os.path.normpath(args.experiment_dir))
        
        out_dir = os.path.join(script_dir, "iqp_solutions", f"{exp_folder_name}_threshold_{args.target_perc}")
        os.makedirs(out_dir, exist_ok=True)
        
        selected_indices = np.where(alpha_result > 0.5)[0]
        for idx in selected_indices:
            src_file = mapping[idx]['path_original_file']
            dst_file = os.path.join(out_dir, mapping[idx]['file'])
            shutil.copy(src_file, dst_file)
        config_id = f"{exp_folder_name}_metric-{args.metric}_mode-{args.mode}_target-{args.target_perc}"
        active_layers = set(m['layer'] for m in mapping)
        save_solution_csv(out_dir, config_id, selected_indices, mapping, active_layers)    
        print(f"\nCopiati {len(selected_indices)} moltiplicatori scelti nella cartella: {out_dir}")

if __name__ == "__main__":
    main()