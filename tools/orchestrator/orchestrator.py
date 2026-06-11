import subprocess
import os
import time
import sys
import argparse
import shutil
import csv
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))
SUBXPAT_DIR = os.path.join(ROOT_DIR, "subxpat")        # CNN_AT/subxpat/
os.environ['PYTHONPATH'] = ROOT_DIR
if SUBXPAT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from subxpat.sxpat.specifications import Specifications
except ImportError as e:
    print(ROOT_DIR)
    print(f"Error importing sxpat specifications: {e}")
    raise
# --- Script and output paths ---
SCRIPT_ANALYZER = os.path.join(CURR_DIR, "npy_generator.py")
ANALYZER_OUTPUT_DIR = os.path.join(CURR_DIR, "experiments/npy_outputs")
SCRIPT_TRAINING = os.path.join(ROOT_DIR, "src/cnn_training.py")

# --- CSV settings ---
CSV_FILE = "results.csv"
csv_lock = Lock()


def get_log_paths(exp_name: str) -> dict:
    """Creates and returns paths for the log files of a given experiment."""
    log_dir = os.path.join(CURR_DIR, "experiments/log", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    return {
        "subxpat":  os.path.join(log_dir, "subxpat.log"),
        "analyzer": os.path.join(log_dir, "analyzer.log"),
        "training": os.path.join(log_dir, "training.log"),
    }


def check_accuracy_in_csv(npy_path: str):
    """
    Checks whether a non-zero accuracy value already exists in the CSV for the given .npy file.
    Returns the accuracy string if found, None otherwise.
    """
    filename_key = os.path.splitext(os.path.basename(npy_path))[0]
    if not os.path.exists(CSV_FILE):
        return None
    with csv_lock:
        with open(CSV_FILE, "r") as f:
            for row in csv.DictReader(f):
                if row["file"] == filename_key:
                    acc = row.get("accuracy", "").strip()
                    if acc and float(acc) != 0:
                        return acc
    return None


def update_accuracy_in_csv(npy_path: str, accuracy: float):
    """Updates the accuracy field for the given .npy file entry in the CSV."""
    filename_key = os.path.splitext(os.path.basename(npy_path))[0]
    if not os.path.exists(CSV_FILE):
        return
    with csv_lock:
        rows = []
        updated = False
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["file"] == filename_key:
                    row["accuracy"] = str(accuracy)
                    updated = True
                rows.append(row)
        if updated:
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def copy_project_to_exp(source: str, destination: str):
    """Copies the SubXPAT project to a fresh experiment directory, excluding git and cache files."""
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))


def is_file_ready(file_path: str) -> bool:
    """Returns True if no process currently has the file open (i.e. it is safe to read)."""
    try:
        subprocess.check_output(["lsof", "-w", file_path])
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def run_analyzer(file_path: str, bitwidth: int, output_dir: str, exp_name: str, log_path: str) -> str:
    """
    Runs the circuit analyzer script on a given Verilog file.
    Returns the path to the generated .npy output file.
    """
    filename = os.path.basename(file_path)
    output_npy_name = os.path.splitext(filename)[0] + f"_{exp_name}.npy"
    output_npy_path = os.path.join(output_dir, output_npy_name)
    cmd = [
        sys.executable, SCRIPT_ANALYZER,
        file_path, str(bitwidth), output_npy_path,
        "--experiment-name", exp_name,
    ]
    with open(log_path, "a") as lfile:
        lfile.write(f"\n--- ANALYZING: {filename} ---\n")
        subprocess.run(cmd, check=True, stdout=lfile, stderr=lfile)
    return output_npy_path


def run_training(npy_path: str, conv_type: str, model_name: str, exact_acc_val, bitwidth: int, log_path: str) -> str:
    """
    Runs the CNN training script on a given .npy file.
    Skips training if a valid accuracy result already exists in the CSV.
    Returns the path to the input .npy file.
    """
    
    filename_key = os.path.splitext(os.path.basename(npy_path))[0]

    if check_accuracy_in_csv(npy_path):
        print(f"Skipping training for '{filename_key}': accuracy already recorded.")
        return npy_path

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(ROOT_DIR,"src") + ":" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, SCRIPT_TRAINING,
        "--conv_type", str(conv_type),
        "--model_name", str(model_name),
        "--input_path", npy_path,
        "--bit_width", str(bitwidth),
    ]

    if exact_acc_val is not None:
        cmd.extend(["--exact_accuracy", str(exact_acc_val)])
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as e:
        print(e.stderr)

    with open(log_path, "a") as lfile:
        lfile.write(f"\n--- TRAINING: {filename_key} ---\n")
        lfile.write(result.stdout)

    # Parse the final accuracy from the training script output
    final_accuracy = None
    for line in result.stdout.splitlines():
        if line.startswith("FINAL_ACCURACY:"):
            try:
                final_accuracy = float(line.split(":")[1].strip())
            except ValueError:
                pass
            break

    if final_accuracy is not None:
        update_accuracy_in_csv(npy_path, final_accuracy)

    return npy_path


def orchestrator(args, subxpat_argv: list):
    """
    Main orchestration loop. Copies the SubXPAT project to an experiment-specific directory,
    launches the SubXPAT generation process, and concurrently runs analysis and training
    on each new Verilog file as it is produced.
    """
    start_timestamp = time.time()
    exp_name = args.experiment_name
    logs = get_log_paths(exp_name)
    print(f"--- Orchestrator started: '{exp_name}' ---")

    exp_subxpat_dir = os.path.join(CURR_DIR,"experiments" ,exp_name)
    copy_project_to_exp(SUBXPAT_DIR, exp_subxpat_dir)

    venv_activate = os.path.join(exp_subxpat_dir, ".venv", "bin", "activate")
    if not os.path.exists(venv_activate):
        subprocess.run(["make", "setup"], cwd=exp_subxpat_dir, check=True)

    # Build and launch the SubXPAT generation command inside the experiment's venv
    subxpat_cmd = " ".join(["python3", "main.py"] + subxpat_argv)
    full_shell_command = f"source {venv_activate} && {subxpat_cmd}"

    try:
        bitwidth = int(int(args.exact_benchmark.split("_i")[1].split("_")[0]) // 2)
    except Exception:
        sys.exit("Error: could not extract bitwidth from benchmark name. Expected format: '..._i<N>_...'")

    os.makedirs(ANALYZER_OUTPUT_DIR, exist_ok=True)
    ver_dir = os.path.join(exp_subxpat_dir, "output", "ver")

    with open(logs["subxpat"], "w") as subxpat_log:
        proc_gen = subprocess.Popen(
            ["/bin/bash", "-c", full_shell_command],
            cwd=exp_subxpat_dir,
            stdout=subxpat_log,
            stderr=subxpat_log,
        )

    processed_files = set()
    training_futures = []

    with ProcessPoolExecutor() as analyzer_executor, ThreadPoolExecutor(max_workers=2) as training_executor:
        while proc_gen.poll() is None or training_futures:
            training_futures = [f for f in training_futures if not f.done()]

            if os.path.exists(ver_dir):
                for filename in os.listdir(ver_dir):
                    full_path = os.path.join(ver_dir, filename)
                    if not filename.endswith(".v"):
                        continue
                    if not os.path.isfile(full_path) or full_path in processed_files:
                        continue
                    if os.path.getmtime(full_path) <= start_timestamp:
                        continue
                    if not is_file_ready(full_path):
                        continue

                    processed_files.add(full_path)
                    future_ana = analyzer_executor.submit(
                        run_analyzer, full_path, bitwidth, ANALYZER_OUTPUT_DIR, exp_name, logs["analyzer"]
                    )

                    def schedule_training(f_ana):
                        """Callback: submits training once analysis is complete."""
                        try:
                            npy_path = f_ana.result()
                            t_fut = training_executor.submit(
                                run_training, npy_path, args.conv_type, args.model_name,
                                args.exact_accuracy, bitwidth, logs["training"]
                            )
                            training_futures.append(t_fut)
                        except Exception as e:
                            print(f"Error scheduling training: {e}")

                    future_ana.add_done_callback(schedule_training)

            time.sleep(1)

    print("--- Orchestration completed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for SubXPAT approximate circuit generation and CNN evaluation.")
    parser.add_argument("--conv-type",        default="3",    help="Convolution type for CNN training.")
    parser.add_argument("--model-name",       default="resnet", help="CNN model name.")
    parser.add_argument("--exact-accuracy",   type=int, default=None, help="Known exact model accuracy (optional).")
    parser.add_argument("--experiment-name",  required=True,  help="Unique name for this experiment run.")
    args, subxpat_argv = parser.parse_known_args()
    # Parse SubXPAT specifications from the remaining argv for validation and benchmark extraction
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + subxpat_argv
    specs = Specifications.parse_args()
    sys.argv = original_argv
    vars(args).update(vars(specs))

    orchestrator(args, subxpat_argv)