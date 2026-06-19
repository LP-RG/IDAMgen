import torch
import numpy as np
import mat_mul
import torch.nn as nn
import torch.optim as optim
import modules.convolution as cc
import models.resnet8 as resnet8
import models.resnet20 as resnet20
import models.lenet5 as lenet5
import models.vgg16 as vgg16
import models.alexnet_cifar10 as alexnet_cifar10
import models.resnet56 as resnet56
import modules.data_loaders as data_loader
import modules.tsne_visualization as tsne_vis
import argparse
import sys
import os
import time
import gc

trained_models_path = "./trained_models/"
device = "cuda"

MODEL_NAME_ALIASES = {
    # file name is resnet20.py, checkpoint/datasets use "resnet"
    "resnet20": "resnet",
    "lenet": "lenet5",
}

def normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    return MODEL_NAME_ALIASES.get(name, name)

# (C, H, W) in PyTorch convention, matching each model's input tensor
MODEL_IMAGE_SHAPES = {
    "lenet5":          (1, 32, 32),
    "resnet":          (3, 32, 32),
    "resnet8":         (3, 32, 32),
    "vgg16":           (3, 32, 32),
    "alexnet_cifar10": (3, 32, 32),
    "resnet56":        (3, 32, 32),
}

MODEL_FACTORIES = {
    "resnet": resnet20.ResNet20,
    "lenet5": lenet5.LeNet5,
    "vgg16": vgg16.VGG16,
    "alexnet_cifar10": alexnet_cifar10.AlexNetCIFAR10,
    "resnet56": resnet56.ResNet56_CIFAR100,
    "resnet8": resnet8.ResNet8
}
train_loader = None
test_loader = None
_classes = None

def set_data_loaders(model_name: str, batch_size: int = 64):
    global train_loader, test_loader, _classes
    name = model_name.lower()

    if name in ("lenet5", "resnet", "resnet8"):
        batch_size = 64
    elif name == "vgg16":
        batch_size = 128
    elif name == "alexnet_cifar10":
        batch_size = 128
    elif name == "resnet56":
        batch_size = 128
    

    train_loader, test_loader, _classes = data_loader.get_datasets(batch_size, model_name)

def get_exact_training_setup(model_name: str, model: nn.Module):
    name = model_name.lower()

    if name == "resnet" or name == "resnet8":
        epochs = 100
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "lenet5":
        epochs = 20
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return epochs, optimizer, scheduler

    if name == "vgg16":
        epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.005, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "alexnet_cifar10":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "resnet56":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    epochs = 100
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return epochs, optimizer, scheduler

def calibration(model, stats=False):
    print("Calibrating model...")
    if stats:
        model.eval()
    else:
        model.train()
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        model(inputs)

def train_one_epoch(epoch, model, optimizer, criterion):
    print(f"Training epoch {epoch + 1}...")
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{batch:>5d}/{len(train_loader):>5d}]")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f"Epoch {epoch + 1}: Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

def test(model):
    print("Testing model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
    return acc

def build_model(model_name: str, conv_type: int, bit_width: int, signed: bool, zone: bool, multiplier_matrix=None, num_classes: int = 10):
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Model '{model_name}' non supportato.")
    return MODEL_FACTORIES[model_name](
        multiplier_matrix,
        num_classes=num_classes,
        conv_type=conv_type,
        bit_width=bit_width,
        signed=signed,
        zone=zone
    ).to(device)

def new_training_method(model_name: str, multiplier_matrix=None, conv_type: int = 1, bit_width: int = 8, signed: bool = False, zone: bool = False, exact_accuracy: float = 0, no_retraining = False):
    input_name = multiplier_matrix.split("/")[-1] if multiplier_matrix is not None else "None"
    print(f"Network training with parameters: model_name = {model_name}, conv_type = {conv_type}, bit_width = {bit_width}, signed = {signed}, input = {input_name}")
    models_dir = trained_models_path.rstrip('/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    exact_path = os.path.join(models_dir, f"{model_name}.pth")
    quant_path = os.path.join(models_dir, f"{model_name}_q{bit_width}.pth")
    #---------------------------------------------------------#
    approx_tag = os.path.splitext(input_name)[0] if multiplier_matrix is not None else "default"
    approx_noretrain_path = os.path.join(
        models_dir, f"{model_name}_a{bit_width}_{approx_tag}_noretrain.pth"
    )
    approx_retrained_best_path = os.path.join(
        models_dir, f"{model_name}_a{bit_width}_{approx_tag}_retrained_best.pth"
    )
    #---------------------------------------------------------#
    num_classes = _classes if _classes else 10
    if conv_type == 1:
        if os.path.exists(exact_path):
            print("Carico modello esatto e avvio test...")
            model = build_model(model_name, conv_type=1, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            model.load_state_dict(torch.load(exact_path, weights_only=True))
            calibration(model)
            return test(model)
        else:
            print("Alleno modello esatto...")
            model = build_model(model_name, conv_type=1, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            epochs, optimizer, scheduler = get_exact_training_setup(model_name, model)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
            torch.save(model.state_dict(), exact_path)
            return test(model)

    if conv_type == 2:
        exact_exists = os.path.exists(exact_path)
        quant_exists = os.path.exists(quant_path)
        if exact_exists and (not quant_exists):
            print("Training quantizzato (5 epoche)...")
            model = build_model(model_name, conv_type=2, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            model.load_state_dict(torch.load(exact_path, weights_only=True), strict=False)
            calibration(model)
            criterion = nn.CrossEntropyLoss()
            lr = 0.001 if bit_width == 4 else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
            best_acc = 0.0
            for epoch in range(5):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
                acc = test(model)
                best_acc = max(best_acc, acc)
            torch.save(model.state_dict(), quant_path)
            return best_acc
        if exact_exists and quant_exists:
            print("Test modello quantizzato...")
            model = build_model(model_name, conv_type=2, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            model.load_state_dict(torch.load(quant_path, weights_only=True))
            calibration(model)
            return test(model)
        raise RuntimeError("Allena prima il modello esatto.")

    if conv_type == 3:
        if not os.path.exists(quant_path):
            raise RuntimeError("Allena prima il modello quantizzato.")
        print("Retrain modello approssimato (3 epoche)...")
        model = build_model(model_name, conv_type=3, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        criterion = nn.CrossEntropyLoss()
        lr = 0.001 if bit_width == 4 else 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
        best_accuracy = 0
        best_state = None
        if(no_retraining):
            acc = test(model)
            torch.save(model.state_dict(), approx_noretrain_path)
            print(f"Saved approximate (no-retrain) checkpoint to: {approx_noretrain_path}")
            return acc
        for epoch in range(3):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
            acc = test(model)
            if(acc < exact_accuracy - 3):
                print("not_good_enough")
                return acc
            if acc > best_accuracy:
                best_accuracy = acc
                # Keep CPU copy of best checkpoint to avoid GPU memory spikes.
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        if best_state is not None:
            torch.save(best_state, approx_retrained_best_path)
        else:
            torch.save(model.state_dict(), approx_retrained_best_path)
        print(f"Saved approximate (retrained-best) checkpoint to: {approx_retrained_best_path}")
        return best_accuracy
    
    if(conv_type == 5):
        model = build_model(model_name, conv_type=5, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        calibration(model,True)
        print("Calibration for stats Done")
        return None
    raise ValueError(f"conv_type={conv_type} non supportato.")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def clean_gpu(model=None, optimizer=None, scheduler=None):
    """Pulisce GPU e variabili non più usate."""
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

# ---------------------------------------------------------- #
def run_tsne_experiment(model_name: str, perplexity: int = 30,
                        max_iter: int = 1000,
                        max_train: int = 2000, max_test: int = 1000,
                        classes=None, seed=42, show_misclassifications: bool = False,
                        feature_space: str = "layer",
                        feature_layers=None,
                        stages=None, tsne_multiplier_paths=None,
                        bit_width: int = 8,
                        save_static: bool = True,
                        save_dash_artifact: bool = True):
    """Load trained CNN checkpoints and visualise misclassifications with t-SNE.

    Parameters
    ----------
    model_name : str
        Must match one of the keys in MODEL_FACTORIES
    perplexity, max_iter : int
        t-SNE hyperparameters forwarded to sklearn.
    max_train, max_test : int
        Maximum number of samples taken from each split for t-SNE (random
        subsample).
    classes : list of int or None
        Subset of class labels to visualise.  None shows all.
    seed : int
        Random seed for subsampling and t-SNE initialisation.
    show_misclassifications : bool
        If True, additionally saves a grid of misclassified images.
        (Requires `image_shape` to be known for the selected model.)
    feature_space : str
        Vector space used as t-SNE input:
        - "layer" uses activations from a hooked module path/alias
        - "pixels" uses flattened raw input pixels
    feature_layers : list[str]
        List of layer alias/paths used when `feature_space="layer"`.
        Supported aliases include: penultimate, logits, conv1, conv2, block1, block2.
        You can also pass explicit `named_modules()` paths (e.g. `layer2.0`).
    stages : list[str] or None
        Which hardware stages to visualise. Allowed values are:
        "exact", "quantized", "approximate".
        None defaults to ["exact"].
    tsne_multiplier_paths : list[str] or str or None
        Path(s) to the .npy multiplier lookup table(s).
        Required when stages contains "approximate".
    bit_width : int
        Bit width used to resolve the quantized checkpoint filename
        (e.g., <model>_q8.pth).
    """
    model_name = normalize_model_name(model_name)

    # _classes is set by set_data_loaders()
    num_classes = _classes if _classes else 10
    exact_path = os.path.join(trained_models_path, f"{model_name}.pth")
    quant_path = os.path.join(trained_models_path, f"{model_name}_q{bit_width}.pth")
    stages = stages or ["exact"]
    image_shape = MODEL_IMAGE_SHAPES.get(model_name.lower())
    resolved_layer_path = None
    # Keep one fixed train/test subset across all selected stages in this run.
    # This makes exact/quantized/approximate embeddings directly comparable.
    subsample_state = {}
    if len(stages) == 1:
        print(
            "[NOTE] Running a single t-SNE stage. Plot is valid for this run, "
            "but cross-run comparisons require fixed seed/sampling/t-SNE params "
            "(seed, max_train, max_test, perplexity, max_iter)."
        )

    def resolve_tsne_layer_path(model: nn.Module, requested_layer: str) -> str:
        """
        Resolve a user-friendly alias into a concrete `named_modules()` path
        for forward-hook feature extraction.
        """
        if requested_layer is None:
            raise ValueError("feature_layer cannot be None when feature_space='layer'")

        modules = dict(model.named_modules())
        if requested_layer in modules:
            return requested_layer

        alias = requested_layer.lower()

        linear_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        conv_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, cc.Conv2d_custom)]
        avg_pool_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.AdaptiveAvgPool2d)]

        if alias == "logits":
            if not linear_modules:
                raise ValueError("Alias 'logits' requires at least one nn.Linear in the model.")
            return linear_modules[-1][0]

        if alias == "penultimate":
            # Usual case: second-to-last linear.
            if len(linear_modules) >= 2:
                return linear_modules[-2][0]
            # ResNet-style: only one linear (fc); use avg_pool output as features before fc.
            if avg_pool_modules:
                return avg_pool_modules[-1][0]
            if linear_modules:
                return linear_modules[0][0]
            raise ValueError("Alias 'penultimate' could not be resolved (no Linear / AdaptiveAvgPool2d found).")

        if alias == "conv1":
            if not conv_modules:
                raise ValueError("Alias 'conv1' requires Conv2d_custom layers in the model.")
            return conv_modules[0][0]

        if alias == "conv2":
            if len(conv_modules) >= 2:
                return conv_modules[1][0]
            if conv_modules:
                return conv_modules[0][0]
            raise ValueError("Alias 'conv2' requires at least one Conv2d_custom layer in the model.")

        if alias == "block1":
            for candidate in ("layer1", "block1", "pool1"):
                if candidate in modules:
                    return candidate
            raise ValueError("Alias 'block1' could not be resolved (expected one of: layer1/block1/pool1).")

        if alias == "block2":
            for candidate in ("layer2", "block2", "pool2"):
                if candidate in modules:
                    return candidate
            raise ValueError("Alias 'block2' could not be resolved (expected one of: layer2/block2/pool2).")

        # Unknown alias: treat as an explicit module path.
        return requested_layer

    import datetime
    import json
    save_dir = "./plots"
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_dir, feature_space, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    if feature_layers is None:
        feature_layers = ["penultimate"]
        
    metadata = {
        "model_name": model_name,
        "feature_space": feature_space,
        "feature_layers": feature_layers,
        "stages": stages,
        "tsne_multiplier_paths": tsne_multiplier_paths,
        "bit_width": bit_width,
        "perplexity": perplexity,
        "max_iter": max_iter,
        "max_train": max_train,
        "max_test": max_test,
        "classes": classes,
        "seed": seed,
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    expanded_stages = []
    for stage in stages:
        if stage == "approximate":
            if not tsne_multiplier_paths:
                raise ValueError("Approximate stage requires at least one --tsne_multiplier_path <path.npy>.")
            paths = [tsne_multiplier_paths] if isinstance(tsne_multiplier_paths, str) else tsne_multiplier_paths
            for path in paths:
                expanded_stages.append((stage, path))
        else:
            expanded_stages.append((stage, None))

    for feature_layer in feature_layers:
        resolved_layer_path = None
        for stage, tsne_multiplier_path in expanded_stages:
            if stage == "exact":
                if not os.path.exists(exact_path):
                    raise FileNotFoundError(
                        f"No exact checkpoint at '{exact_path}'. "
                        f"Train first with --model_name {model_name} --conv_type 1."
                    )
                model = build_model(model_name, conv_type=1, bit_width=bit_width,
                                    signed=False, zone=False, multiplier_matrix=None,
                                    num_classes=num_classes)
                model.load_state_dict(torch.load(exact_path, weights_only=True))
                output_tag = "exact"
            elif stage == "quantized":
                if not os.path.exists(quant_path):
                    raise FileNotFoundError(
                        f"No quantized checkpoint at '{quant_path}'. "
                        f"Train first with --model_name {model_name} --conv_type 2."
                    )
                model = build_model(model_name, conv_type=2, bit_width=bit_width,
                                    signed=False, zone=False, multiplier_matrix=None,
                                    num_classes=num_classes)
                model.load_state_dict(torch.load(quant_path, weights_only=True))
                output_tag = "quantized"
            elif stage == "approximate":
                if not os.path.exists(tsne_multiplier_path):
                    raise FileNotFoundError(
                        f"Approximate multiplier table not found: '{tsne_multiplier_path}'."
                    )
                model = build_model(model_name, conv_type=3, bit_width=bit_width,
                                    signed=False, zone=False,
                                    multiplier_matrix=tsne_multiplier_path,
                                    num_classes=num_classes)
                approx_tag = os.path.splitext(os.path.basename(tsne_multiplier_path))[0]
                approx_retrained_best_path = os.path.join(
                    trained_models_path, f"{model_name}_a{bit_width}_{approx_tag}_retrained_best.pth"
                )
                if os.path.exists(approx_retrained_best_path):
                    print(f"Loading retrained approximate checkpoint: {approx_retrained_best_path}")
                    model.load_state_dict(
                        torch.load(approx_retrained_best_path, weights_only=True),
                        strict=False
                    )
                else:
                    print(
                        f"[WARN] Retrained approximate checkpoint not found for '{approx_tag}'. "
                        f"Falling back to quantized checkpoint: {quant_path}"
                    )
                    if not os.path.exists(quant_path):
                        raise FileNotFoundError(
                            f"No available checkpoints for approximate stage:\n"
                            f"- retrained: '{approx_retrained_best_path}'\n"
                            f"- quantized base: '{quant_path}'.\n"
                            f"Train conv_type=2 first and (optionally) conv_type=3 retrained-best for this multiplier."
                        )
                    model.load_state_dict(torch.load(quant_path, weights_only=True), strict=False)
                output_tag = f"approximate_{approx_tag}"
            else:
                raise ValueError(
                    f"Unknown stage '{stage}'. Use one of: exact, quantized, approximate."
                )
    
            # Quantized/approximate conv paths require observer-derived quantization
            # parameters (activation_scale, zero-points, etc.). Ensure they exist
            # before inference-time feature extraction for t-SNE.
            if stage in ("quantized", "approximate"):
                calibration(model)
    
            model.to(device)
            if feature_space == "layer" and resolved_layer_path is None:
                resolved_layer_path = resolve_tsne_layer_path(model, feature_layer)
                print(f"Using layer alias/path '{feature_layer}' resolved to '{resolved_layer_path}'.")
            print(f"\n--- Running t-SNE for stage: {stage} ({feature_space}, layer: {feature_layer}) ---")
            tsne_vis.run_tsne_cnn_experiment(
                model, train_loader, test_loader, device,
                model_name=model_name,
                perplexity=perplexity,
                max_iter=max_iter,
                max_train=max_train,
                max_test=max_test,
                classes=classes,
                seed=seed,
                show_misclassifications=show_misclassifications,
                image_shape=image_shape,
                feature_space=feature_space,
                output_tag=output_tag,
                feature_layer_path=resolved_layer_path,
                feature_layer_requested=feature_layer,
                save_static=save_static,
                save_dash_artifact=save_dash_artifact,
                subsample_state=subsample_state,
                run_id=run_id,
            )

# ---------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with simplified logic and model_name routing.")
    parser.add_argument("--model_name", type=str, default="resnet")
    parser.add_argument("--conv_type", type=int, default=1)
    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--signed", action="store_true", default=False)
    parser.add_argument("--zone", action="store_true", default=False)
    parser.add_argument("--input_path", nargs='?', default=None)
    parser.add_argument("--exact_accuracy", type=float, default=0)
    parser.add_argument("--no_retraining", action="store_true", default=False)
    # ---------------------------------------------------------- #
    parser.add_argument("--tsne", action="store_true", default=False,
                        help="Visualise input-space t-SNE with CNN misclassifications.")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--tsne_max_train", type=int, default=2000)
    parser.add_argument("--tsne_max_test", type=int, default=1000)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--tsne_classes", type=int, nargs="+", default=None,
                        metavar="C", help="Classes to visualise, e.g. --tsne_classes 5 8")
    parser.add_argument("--show_misclassifications", action="store_true", default=False,
                        help="After t-SNE, display a grid of CNN misclassified images.")
    parser.add_argument("--tsne_feature_space", type=str, default="layer",
                        choices=["layer", "pixels"],
                        help="Feature space for t-SNE: arbitrary layer hook or raw pixels.")
    parser.add_argument("--tsne_feature_layer", type=str, nargs="+", default=["penultimate"],
                        help=("Layer alias/path(s) used when --tsne_feature_space layer. "
                              "Aliases: penultimate, logits, conv1, conv2, block1, block2 "
                              "(or explicit module paths like layer2.0)."))
    parser.add_argument("--tsne_stages", type=str, nargs="+", default=["exact"],
                        choices=["exact", "quantized", "approximate"],
                        help="Stages to visualise: exact, quantized, approximate.")
    parser.add_argument("--tsne_multiplier_path", type=str, nargs="+", default=None,
                        help="Required for approximate stage: one or more .npy multiplier table paths.")
    parser.add_argument("--tsne-no-save-static", action="store_false", dest="tsne_save_static",
                        help="Do not save static t-SNE/misclassification PNG outputs.")
    parser.add_argument("--tsne-no-save-dash-artifact", action="store_false", dest="tsne_save_dash_artifact",
                        help="Do not save Dash .npz artifacts.")
    parser.set_defaults(tsne_save_static=True, tsne_save_dash_artifact=True)
    # ---------------------------------------------------------- #
    args = parser.parse_args()

    device = "cuda"
    results = {}
    start = time.time()
    p = args.input_path

    # ---------------------------------------------------------- #
    if args.tsne:
        setup_seed(args.tsne_seed)
        set_data_loaders(args.model_name)
        run_tsne_experiment(
            args.model_name,
            perplexity=args.tsne_perplexity,
            max_iter=args.tsne_max_iter,
            max_train=args.tsne_max_train,
            max_test=args.tsne_max_test,
            classes=args.tsne_classes,
            seed=args.tsne_seed,
            show_misclassifications=args.show_misclassifications,
            feature_space=args.tsne_feature_space,
            feature_layers=args.tsne_feature_layer,
            stages=args.tsne_stages,
            tsne_multiplier_paths=args.tsne_multiplier_path,
            bit_width=args.bit_width,
            save_static=args.tsne_save_static,
            save_dash_artifact=args.tsne_save_dash_artifact,
        )
        sys.exit(0)

    # ---------------------------------------------------------- #
    if p is None:
        setup_seed(42)
        set_data_loaders(args.model_name)  
        acc = new_training_method(
            args.model_name,
            None,
            args.conv_type,
            args.bit_width,
            args.signed,
            args.zone,
            args.exact_accuracy
        )
        print(acc)
        sys.exit(0)

    if not os.path.exists(p):
        print(f"Error: The input path '{p}' does not exist.")
        sys.exit(1)

    # Se p è un singolo file
    if os.path.isfile(p):
        setup_seed(42)
        set_data_loaders(args.model_name)
        acc = new_training_method(
            args.model_name,
            p,
            args.conv_type,
            args.bit_width,
            args.signed,
            args.zone,
            args.exact_accuracy,
            args.no_retraining
        )
        print(f"FINAL_ACCURACY:{acc}")
        clean_gpu()
        sys.exit(0)
    # Se p è una cartella
    for f in os.listdir(p):
        if not f.endswith(".npy"):
            continue

        file_path = os.path.join(p, f)

        setup_seed(42)

        set_data_loaders(args.model_name)

        acc = new_training_method(
            args.model_name,
            file_path,
            args.conv_type,
            args.bit_width,
            args.signed,
            args.zone,
            args.exact_accuracy,
            args.no_retraining
        )
        print(f"FINAL_ACCURACY:{acc}")
        results[f] = acc
        clean_gpu()

    print(results)
    print(f"Total training time: {time.time() - start}")
