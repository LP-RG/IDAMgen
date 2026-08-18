import argparse
import json
import numpy as np
import cvxpy as cp
import os
import glob
import shutil

# Total multiplications for each layer
LAYER_MULTS = {
    "1_1": 4.43e+10,
    "1_2": 3.92e+10,
    "2_1": 3.57e+10,
    "2_2": 5.08e+10,
    "2_s": 3.20e+09,
    "3_1": 2.80e+10,
    "3_2": 3.44e+10,
    "3_s": 3.14e+09,
    "s_8": 1.14e+10,
}

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

def load_weighted_costs(mapping, metric="area"):
    """
    Extracts unit costs from the JSON mapping and calculates weighted costs by number of operations.
    Calculates the baseline cost using the EXACT_DATA global dictionary.
    """
    
    if metric not in EXACT_DATA:
        raise KeyError(f"Metric '{metric}' not found in EXACT_DATA. Please add it to the global dictionary.")
    
    exact_unit_val = EXACT_DATA[metric]

    weighted_costs = np.zeros(len(mapping))
    baseline_cost = 0.0
    
    active_layers = set([m['layer'] for m in mapping])
    
    for layer_str in active_layers:
        baseline_cost += exact_unit_val * LAYER_MULTS[layer_str]

    for i, item in enumerate(mapping):
        layer_str = item['layer']
        
        unit_val = item.get(metric)
        if unit_val is None:
            raise ValueError(f"Metric '{metric}' not found for file: {item['file']}")
            
        weighted_costs[i] = unit_val * LAYER_MULTS[layer_str]
    
    return weighted_costs, baseline_cost

def make_matrix_psd(G):
    """Projects the matrix to make it positive semi-definite (PSD)."""
    es, us = np.linalg.eig(G)
    es[es < 0] = 0
    G_psd = us @ np.diag(es) @ us.T
    G_psd = (G_psd + G_psd.T) / 2
    return G_psd

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
    
    try:
        prob.solve(solver=solver, verbose=False, TimeLimit=120)
    except Exception as e:
        print(f"Solver error {solver}: {e}. Falling back to default.")
        prob.solve(verbose=False)

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
    
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment folder containing the _matrix.npy and _map.json")
    parser.add_argument("--metric", type=str, default="area", help="Hardware metric to optimize (e.g., area, power, delay, pa, pda)")
    
    parser.add_argument("--mode", type=str, default='min_loss', choices=['min_loss', 'min_area'])
    parser.add_argument("--target_perc", type=float, required=True, help="Target percentage (e.g., 0.8 for 80% of baseline budget in min_loss, or absolute threshold in min_area)")
    parser.add_argument("--solver", type=str, default="GUROBI", help="Solver to use")

    args = parser.parse_args()

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
            
        print(f"\nCopiati {len(selected_indices)} moltiplicatori scelti nella cartella: {out_dir}")

if __name__ == "__main__":
    main()