from vpadanalyzer.synthesis import Synthesis
import os
import sys
import sub_xpat_circuits_generator 
import multiplier_outputs_plotting 
import sub_x_pat_simulator 
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILE = os.path.join(SCRIPT_DIR, "sub_x_pat_multiplier.py")
CSV_FILENAME = "circuits_area_power.csv"
CSV_FIELDS = ["file", "area", "power", "delay", "pda", "mean_ae", "mean_ae_cnn", "max_ae"]

PATH_TO_LOCAL_OPEN_STA = 'PLACEHOLDER'


def patch_opensta_path(path: str):
    if(os.path.isfile(path)):
        import vpadanalyzer.paths
        vpadanalyzer.paths.OPENSTA = path   
        import vpadanalyzer.synthesis
        vpadanalyzer.synthesis.OPENSTA = path
    
    



def circuits_analizer(input_path):
    """Analyzes a circuit and returns its area, power, and delay."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: '{input_path}'")
    area = Synthesis.area(input_path)
    power = Synthesis.power(input_path)
    delay = Synthesis.delay(input_path)
    print(f"Synthesis completed: {os.path.basename(input_path)}")
    return {"file": os.path.basename(input_path), "area": area, "power": power, "delay": delay}


def create_matrices(multipliers_folder, bitwidth, output_plot_path):
    """Simulates all multipliers from existing .npy files and returns error metrics."""
    if not os.path.isdir(multipliers_folder):
        raise NotADirectoryError(f"Folder not found: '{multipliers_folder}'")
    os.makedirs(output_plot_path, exist_ok=True)
    results = []
    for filename in sorted(os.listdir(output_plot_path)):
        if not filename.endswith(".v"):
            continue
        npy_path = os.path.join(output_plot_path, filename.split(".")[0])
        name = filename.split(".")[0]
        input_path = os.path.join(multipliers_folder, name + ".v")
        if not os.path.isfile(input_path):
            print(f"Warning: no matching .v file for '{filename}', skipping.")
            continue
        print(f"Simulating multiplier: {filename}")
        try:
            sub_xpat_circuits_generator.generate_approx_mult_function(input_path, bitwidth)
            mean_ae, mean_ae_cnn, max_error = sub_x_pat_simulator.execute_save(bitwidth, npy_path)
            results.append({"file": name + ".v", "mean_ae": mean_ae, "mean_ae_cnn": mean_ae_cnn, "max_ae": max_error})
            multiplier_outputs_plotting.plots(name, npy_path + ".npy", output_plot_path)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return results


def analyze_multipliers(multipliers_folder):
    """Analyzes all Verilog multiplier files and returns area, power, delay, and PDA metrics."""
    if not os.path.isdir(multipliers_folder):
        raise NotADirectoryError(f"Folder not found: '{multipliers_folder}'")
    results = []
    for filename in sorted(os.listdir(multipliers_folder)):
        if not filename.endswith(".v"):
            continue
        input_path = os.path.join(multipliers_folder, filename)
        if not os.path.isfile(input_path):
            continue
        try:
            data = circuits_analizer(input_path)
            data["pda"] = data["area"] * data["power"] * data["delay"]
            results.append(data)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return results


def merge_results(results, mean_ae_results):
    """Merges synthesis results and error metrics on the 'file' field."""
    ae_dict = {item["file"]: item for item in mean_ae_results}
    merged = []
    for r in results:
        if r["file"] not in ae_dict:
            print(f"Warning: '{r['file']}' not found in simulation results.")
        merged.append({**r, **ae_dict.get(r["file"], {})})
    return merged


def write_csv(merged_results, output_path):
    """Writes merged results to a CSV file."""
    output_file = os.path.join(output_path, CSV_FILENAME)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged_results)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise NotImplementedError("Usage: script.py <multipliers_folder> <bitwidth> <output_path>")
    
    patch_opensta_path(PATH_TO_LOCAL_OPEN_STA)

    multipliers_folder, bitwidth, output_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    results = analyze_multipliers(multipliers_folder)
    mean_ae_results = create_matrices(multipliers_folder, bitwidth, output_path)

    if os.path.exists(GENERATED_FILE):
        os.remove(GENERATED_FILE)
    else:
        print(f"Warning: generated file not found: '{GENERATED_FILE}'")

    if results:
        for data in results:
            print(data)
    if mean_ae_results:
        print(mean_ae_results)

    merged_results = merge_results(results, mean_ae_results)
    write_csv(merged_results, output_path)