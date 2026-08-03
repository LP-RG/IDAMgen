from __future__ import annotations

import torch
import torch.nn as nn
import argparse
import os
import sys
import datetime
import json
import tsne_visualization as tsne_vis
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))
SRC_PATH = os.path.join(ROOT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import modules.data_loaders as data_loader
import modules.convolution as cc
from cnn_training import new_training_method
from modules.common import (
    trained_models_path, device,
    normalize_model_name, MODEL_IMAGE_SHAPES, build_model, setup_seed,
    train_loader, test_loader, _classes,
)


def calibration(model, stats=False):
    """Calibrates model activations/weights using the training set."""
    print("Calibrating model...")
    if stats:
        model.eval()
    else:
        model.train()
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        model(inputs)


def set_data_loaders(model_name: str, batch_size: int = 64):
    """Sets appropriate batch sizes based on the model architecture and loads data."""
    global train_loader, test_loader, _classes
    name = model_name.lower()

    if name in ("lenet5", "resnet", "resnet8"):
        batch_size = 64
    elif name in ("vgg16", "alexnet_cifar10", "resnet56"):
        batch_size = 128

    train_loader, test_loader, _classes = data_loader.get_datasets(batch_size, model_name)


def resolve_tsne_layer_path(model: nn.Module, requested_layer: str) -> str:
    """Resolve a user-friendly alias into a concrete named_modules() path."""
    if requested_layer is None:
        raise ValueError("feature_layer cannot be None when feature_space='layer'")

    modules = dict(model.named_modules())
    if requested_layer in modules:
        return requested_layer

    alias = requested_layer.lower()
    linear_modules  = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    conv_modules    = [(n, m) for n, m in model.named_modules() if isinstance(m, cc.Conv2d_custom)]
    avg_pool_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.AdaptiveAvgPool2d)]

    if alias == "logits":
        if not linear_modules:
            raise ValueError("Alias 'logits' requires at least one nn.Linear in the model.")
        return linear_modules[-1][0]

    if alias == "penultimate":
        if len(linear_modules) >= 2:
            return linear_modules[-2][0]
        if avg_pool_modules:
            return avg_pool_modules[-1][0]
        if linear_modules:
            return linear_modules[0][0]
        raise ValueError("Alias 'penultimate' could not be resolved.")

    if alias == "conv1":
        if not conv_modules:
            raise ValueError("Alias 'conv1' requires Conv2d_custom layers in the model.")
        return conv_modules[0][0]

    if alias == "conv2":
        if len(conv_modules) >= 2:
            return conv_modules[1][0]
        if conv_modules:
            return conv_modules[0][0]
        raise ValueError("Alias 'conv2' requires at least one Conv2d_custom layer.")

    if alias == "block1":
        for candidate in ("layer1", "block1", "pool1"):
            if candidate in modules:
                return candidate
        raise ValueError("Alias 'block1' could not be resolved (expected: layer1/block1/pool1).")

    if alias == "block2":
        for candidate in ("layer2", "block2", "pool2"):
            if candidate in modules:
                return candidate
        raise ValueError("Alias 'block2' could not be resolved (expected: layer2/block2/pool2).")

    # Unknown alias: treat as explicit module path
    return requested_layer


def ensure_checkpoints(model_name: str, bit_width: int, stages: list,
                       tsne_multiplier_paths=None):
    """Train any missing checkpoints required by the requested stages.

    Training order is always exact → quantized → approximate, so that each
    stage has its prerequisite available even when all three are requested at
    once.
    """
    exact_path = os.path.join(trained_models_path, f"{model_name}.pth")
    quant_path  = os.path.join(trained_models_path, f"{model_name}_q{bit_width}.pth")

    needs_exact = any(s in stages for s in ("exact", "quantized", "approximate"))
    needs_quant = any(s in stages for s in ("quantized", "approximate"))
    needs_approx = "approximate" in stages

    if needs_exact and not os.path.exists(exact_path):
        print(f"[train-if-missing] Exact checkpoint not found. Training conv_type=1...")
        new_training_method(model_name, multiplier_matrix=None, conv_type=1,
                            bit_width=bit_width)

    if needs_quant and not os.path.exists(quant_path):
        print(f"[train-if-missing] Quantized checkpoint not found. Training conv_type=2...")
        new_training_method(model_name, multiplier_matrix=None, conv_type=2,
                            bit_width=bit_width)

    if needs_approx:
        paths = ([tsne_multiplier_paths] if isinstance(tsne_multiplier_paths, str)
                 else (tsne_multiplier_paths or []))
        for mpath in paths:
            approx_tag = os.path.splitext(os.path.basename(mpath))[0]
            approx_path = os.path.join(
                trained_models_path,
                f"{model_name}_a{bit_width}_{approx_tag}_retrained_best.pth"
            )
            if not os.path.exists(approx_path):
                print(f"[train-if-missing] Approximate checkpoint not found for '{approx_tag}'. "
                      f"Training conv_type=3...")
                new_training_method(model_name, multiplier_matrix=mpath, conv_type=3,
                                    bit_width=bit_width)


def run_tsne_experiment(model_name: str, perplexity: int = 30, components: str = "2D",
                        max_iter: int = 1000, max_train: int = 2000, max_test: int = 1000,
                        classes=None, seed: int = 42,
                        show_misclassifications: bool = False,
                        feature_space: str = "layer", feature_layers=None,
                        stages=None, tsne_multiplier_paths=None,
                        bit_width: int = 8,
                        save_static: bool = True, save_dash_artifact: bool = True,
                        train_if_missing: bool = False):
    model_name = normalize_model_name(model_name)
    num_classes = _classes if _classes else 10

    exact_path = os.path.join(trained_models_path, f"{model_name}.pth")
    quant_path = os.path.join(trained_models_path, f"{model_name}_q{bit_width}.pth")

    stages = stages or ["exact"]

    if train_if_missing:
        ensure_checkpoints(model_name, bit_width, stages, tsne_multiplier_paths)
    image_shape = MODEL_IMAGE_SHAPES.get(model_name.lower())

    if feature_layers is None:
        feature_layers = ["penultimate"]

    if len(stages) == 1:
        print(
            "[NOTE] Running a single t-SNE stage. Cross-run comparisons require fixed "
            "seed/sampling/t-SNE params (seed, max_train, max_test, perplexity, max_iter)."
        )

    # Persist one subsample across stages so embeddings are directly comparable
    subsample_state = {}

    # Set up output directory
    save_dir = os.path.join(CURR_DIR, "plots")
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_dir, feature_space, model_name, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "feature_space": feature_space,
        "feature_layers": feature_layers,
        "stages": stages,
        "tsne_multiplier_paths": tsne_multiplier_paths,
        "bit_width": bit_width,
        "perplexity": perplexity,
        "components": components,        
        "max_iter": max_iter,
        "max_train": max_train,
        "max_test": max_test,
        "classes": classes,
        "seed": seed,
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Expand "approximate" entries into one entry per multiplier path
    expanded_stages = []
    for stage in stages:
        if stage == "approximate":
            if not tsne_multiplier_paths:
                raise ValueError("Approximate stage requires at least one multiplier path.")
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
                        torch.load(approx_retrained_best_path, weights_only=True), strict=False
                    )
                else:
                    print(
                        f"[WARN] Retrained approximate checkpoint not found for '{approx_tag}'. "
                        f"Falling back to quantized checkpoint: {quant_path}"
                    )
                    if not os.path.exists(quant_path):
                        raise FileNotFoundError(
                            f"No available checkpoints for approximate stage:\n"
                            f"  retrained: '{approx_retrained_best_path}'\n"
                            f"  quantized base: '{quant_path}'.\n"
                            f"Train conv_type=2 first."
                        )
                    model.load_state_dict(torch.load(quant_path, weights_only=True), strict=False)
                output_tag = f"approximate_{approx_tag}"

            else:
                raise ValueError(f"Unknown stage '{stage}'. Use: exact, quantized, approximate.")

            if stage in ("quantized", "approximate"):
                calibration(model)

            model.to(device)

            if feature_space == "layer" and resolved_layer_path is None:
                resolved_layer_path = resolve_tsne_layer_path(model, feature_layer)
                print(f"Layer alias '{feature_layer}' resolved to '{resolved_layer_path}'.")

            print(f"\n--- t-SNE stage: {stage} ({feature_space}, layer: {feature_layer}) ---")
            tsne_vis.run_tsne_cnn_experiment(
                model, train_loader, test_loader, device,
                model_name=model_name,
                perplexity=perplexity,
                components=components,
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


def run_tsne_sweep_experiment(model_name: str, train_schedule: list[int], test_size: int = 1000,
                            perplexity: int = 30, components: str = "2D", max_iter: int = 1000, 
                            seed: int = 42, feature_layer: str = "penultimate", bit_width: int = 8,
                            save_artifact: bool = False, train_if_missing: bool = False, n_repeats: int = 5):
    model_name = normalize_model_name(model_name)
    num_classes = _classes if _classes else 10

    exact_path = os.path.join(trained_models_path, f"{model_name}.pth")

    if train_if_missing:
        ensure_checkpoints(model_name, bit_width, ["exact"])
    image_shape = MODEL_IMAGE_SHAPES.get(model_name.lower())

    if not os.path.exists(exact_path):
        raise FileNotFoundError(
            f"No exact checkpoint at '{exact_path}'. "
            f"Train first with --model_name {model_name} --conv_type 1."    
        )
    model = build_model(model_name, conv_type=1, bit_width=bit_width,
                        signed=False, zone=False, multiplier_matrix=None,
                        num_classes=num_classes)
    model.load_state_dict(torch.load(exact_path, weights_only=True))
    model.to(device)

    feature_layer_path = resolve_tsne_layer_path(model, feature_layer)
    print(f"Layer alias '{feature_layer}' resolved to '{feature_layer_path}'.")

    # Set up output directory
    save_dir = os.path.join(CURR_DIR, "plots")
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_dir, "layer", model_name, "sweep", run_id)
    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "feature_space": "layer",
        "feature_layer": feature_layer,
        "feature_layer_path": feature_layer_path,
        "stages": "exact",
        "bit_width": bit_width,
        "perplexity": perplexity,
        "components": components,        
        "max_iter": max_iter,
        "train_schedule": train_schedule,
        "test_size": test_size,
        "seed": seed,
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
    }    
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    tsne_vis.run_tsne_sweep_cnn_experiment(model, train_loader, test_loader, device, train_schedule,
        model_name=model_name, perplexity=perplexity, components=components,
        max_iter=max_iter, test_size=test_size,
        save_dir=save_dir, seed=seed, image_shape=image_shape,
        feature_layer_path=feature_layer_path,
        feature_layer_requested=feature_layer,
        save_artifact=save_artifact,
        run_id=run_id, run_dir=run_dir, n_repeats=n_repeats)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualisation of CNN feature spaces.")
    parser.add_argument("--model_name", type=str, default="resnet")
    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--tsne_max_train", type=int, default=2000)
    parser.add_argument("--tsne_max_test", type=int, default=1000)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--tsne_classes", type=int, nargs="+", default=None,
                        metavar="C", help="Classes to visualise, e.g. --tsne_classes 5 8")
    parser.add_argument("--show_misclassifications", action="store_true", default=False)
    parser.add_argument("--tsne_feature_space", type=str, default="layer",
                        choices=["layer", "pixels"])
    parser.add_argument("--tsne_feature_layer", type=str, nargs="+", default=["penultimate"],
                        help="Layer alias/path(s): penultimate, logits, conv1, conv2, block1, block2, "
                             "or explicit module paths like layer2.0")
    parser.add_argument("--tsne_stages", type=str, nargs="+", default=["exact"],
                        choices=["exact", "quantized", "approximate"])
    parser.add_argument("--tsne_multiplier_path", type=str, nargs="+", default=None,
                        help="One or more .npy multiplier table paths (required for approximate stage).")
    parser.add_argument("--tsne-no-save-static", action="store_false", dest="tsne_save_static")
    parser.add_argument("--tsne-no-save-dash-artifact", action="store_false", dest="tsne_save_dash_artifact")
    parser.add_argument("--train-if-missing", action="store_true", default=False,
                        help="Automatically train missing checkpoints before running t-SNE.")
    parser.add_argument("--tsne_components", type=str, default="2D", choices=["2D", "3D", "2D+3D"])    
    
    sweep_group = parser.add_argument_group("Sweep (exploratory)")
    sweep_group.add_argument("--tsne_sweep", action="store_true")
    sweep_group.add_argument("--tsne_sweep_train_sizes", type=int, nargs="+")
    sweep_group.add_argument("--tsne_sweep_test_size", type=int, default=1000)
    sweep_group.add_argument("--tsne_sweep_save_dash_artifact", action="store_true")
    sweep_group.add_argument("--tsne_sweep_feature_layer", type=str, default="penultimate", 
                        help="Layer alias/path(s): penultimate, logits, conv1, conv2, block1, block2, "
                             "or explicit module paths like layer2.0")
    sweep_group.add_argument("--tsne_sweep_n_repeats", type=int, default=5)

    parser.set_defaults(tsne_save_static=True, tsne_save_dash_artifact=True)
    args = parser.parse_args()

    setup_seed(args.tsne_seed)
    set_data_loaders(args.model_name)

    if args.tsne_sweep:
        run_tsne_sweep_experiment(
            args.model_name,
            train_schedule=args.tsne_sweep_train_sizes,
            test_size=args.tsne_sweep_test_size,
            perplexity=args.tsne_perplexity,
            components=args.tsne_components,
            max_iter=args.tsne_max_iter,
            seed=args.tsne_seed,
            feature_layer=args.tsne_sweep_feature_layer,
            bit_width=args.bit_width,
            save_artifact=args.tsne_sweep_save_dash_artifact,
            train_if_missing=args.train_if_missing,
            n_repeats=args.tsne_sweep_n_repeats
        )

    else:
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
            train_if_missing=args.train_if_missing,
            components=args.tsne_components   
        )