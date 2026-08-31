import argparse
import os
import json

def parse_header(line):
    # matches: "Network training with parameters: model_name=..., ..."
    prefix = "Network training with parameters:"
    if not line.startswith(prefix):
        return None
    half = line.split(":", 1)[1].strip()
    metadata = {
        "model_name": None,
        "conv_type": None,
        "bit_width": None,
        "signed": None,
        "input": None
    }
    for part in half.split(", "):
        key, value = part.split("=")
        if key in metadata:
            metadata[key] = value
    return metadata


def epoch_marker(line):
    # matches: "Epoch 1" -> 1
    if not line.startswith("Epoch ") or ":" in line:
        return None
    num = line.split(" ")[1]
    return num


def epoch_summary(line):
    # matches: "Epoch 1: Loss: ..., Accuracy: ..."
    if not line.startswith("Epoch ") or ":" not in line:
        return None
    epoch_part, rest = line.split(": ", 1)
    epoch_num = int(epoch_part.split(" ")[1])
    epoch_record = {
        "epoch": epoch_num,
        "train_loss": None,
        "train_acc": None,
        "test_acc": None
    }
    for part in rest.split(", "):
        key, value = part.split(": ")
        if key == "Loss":
            epoch_record["train_loss"] = float(value)
        elif key == "Accuracy":
            epoch_record["train_acc"] = float(value.rstrip("%"))
    return epoch_record


def test_accuracy(line):
    # matches: "Test Accuracy: ..."
    prefix = "Test Accuracy"
    if not line.startswith(prefix):
        return None
    value = line[len(prefix)+2:].strip().rstrip("%")
    return float(value)


def final_accuracy(line):
    # matches:"Exact model accuracy: ..." or "FINAL_ACCURACY: ..."
    if line.startswith("Exact model accuracy: "):
        value = line[len("Exact model accuracy: "):]
    elif line.startswith("FINAL_ACCURACY:"):
        value = line[len("FINAL_ACCURACY:"):]
    else:
        return None
    return float(value.strip())


def parse_log(path):
    """Parse one cnn_training.py log file into {metadata, epochs, final_accuracy}."""
    metadata = {}
    epochs = []
    current = None
    final_acc = None

    with open(path) as fhandle:
        for line in fhandle:
            line = line.strip()
            
            header = parse_header(line)
            if header is not None:
                metadata = header
                continue

            marker = epoch_marker(line)
            if marker is not None:
                current = {
                    "epoch": int(marker),
                    "train_loss": None,
                    "train_acc": None,
                    "test_acc": None
                }
                epochs.append(current)
                continue
            
            summary = epoch_summary(line)
            if summary is not None:
                current["train_loss"] = summary["train_loss"]
                current["train_acc"] = summary["train_acc"]
                continue

            tst_acc = test_accuracy(line)
            # keeps conv_type-1 test lines from attaching to the last-trained epoch
            if tst_acc is not None and (metadata.get("conv_type") in ("2", "3")) and current is not None:
                current["test_acc"] = tst_acc
                continue
            
            final = final_accuracy(line)
            if final is not None:
                final_acc = final
                continue
    
    return {
        "metadata": metadata,
        "epochs": epochs,
        "final_accuracy": final_acc
    }


# ------------------------------------------------------------------ #
if __name__ == "__main__":  # run block when file is executed directly, not when it's imported
    parser = argparse.ArgumentParser(description="Parse a cnn_training.py log into structured JSON.")
    # positional argument 
    parser.add_argument("log_path", type=str, help="Path to the .log file to parse")
    parser.add_argument("--output", type=str, default=None, help="Where to write the JSON. Defaults to parsed_logs/<same name>.json.")
    # reads whatever was typed on the command line and gives you back an object
    args = parser.parse_args()

    result = parse_log(args.log_path)

    if args.output: # did user specify where to write the file?
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.log_path))[0]
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsed_logs")
        os.makedirs(output_dir, exist_ok=True)  # creates that folder if it's not already there
        output_path = os.path.join(output_dir, base_name + ".json") # gives folder and new filename together

    with open(output_path, "w") as f:   # "w" creates file if it doesnt exist, overwrites it if it does
        json.dump(result, f, indent=2)

    print(f"Wrote parsed log to {output_path}")