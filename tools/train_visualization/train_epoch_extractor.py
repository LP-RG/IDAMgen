import argparse
import os
import sys
import torch
import numpy as np
from sklearn.manifold import TSNE

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))
SRC_PATH = os.path.join(ROOT_DIR, "src")
TSNE_VIS_PATH = os.path.join(ROOT_DIR, "tools", "tsne_visualization")
for p in (SRC_PATH, TSNE_VIS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

import cnn_training
from cnn_training import get_exact_training_setup, set_data_loaders, calibration
from train_utils import (save_epoch_features, save_epoch_artifact, save_train_manifest)
from tsne_runner import resolve_tsne_layer_path
from tsne_visualization import _collect_layer_features
from modules.common import (build_model, device, test_loader, trained_models_path,
                            normalize_model_name, _classes)

"""Script function: turning folder of per epoch checkpoints into a folder of per epoch 3D point clouds."""

def resolve_stage(conv_type, bit_width, multiplier_matrix, exact_epochs):
    """Derive stage_tag, per epoch checkpoint filename template, and epoch count
    all together from conv_type."""
    if conv_type == 1:
        return "exact", "{model_name}_epoch{epoch}.pth", exact_epochs
    if conv_type == 2:
        return f"quant_q{bit_width}", "{model_name}_q" + str(bit_width) + "_epoch{epoch}.pth", 5
    if conv_type == 3:
        approx_tag = os.path.splitext(os.path.basename(multiplier_matrix))[0] if multiplier_matrix else "default"
        template = "{model_name}_a" + str(bit_width) + f"_{approx_tag}" + "_epoch{epoch}.pth"
        return f"approx_a{bit_width}_{approx_tag}", template, 3
    raise ValueError(f"conv_type={conv_type} is not supported for extraction.")


def compute_epoch_embedding(X_feat, perplexity=30, max_iter=1000, seed=42, prev_embedding=None):
    """Independent tSNE fit per epoch."""
    if prev_embedding is not None:
        prev_embedding = np.asarray(prev_embedding, dtype=np.float64)
        scale = 1e-4 / (prev_embedding[:, 0].std() + 1e-12)
        init = prev_embedding * scale
    else:
        init = "pca"

    return TSNE(n_components=3, perplexity=perplexity, n_iter=max_iter,
                init=init, random_state=seed).fit_transform(X_feat)


def main():
    """CLI entry point: extract per epoch features and warm started t-SNE embeddings for one training stage."""
    parser = argparse.ArgumentParser(description="Extract per-epoch activations + t-SNE embeddings.")
    parser.add_argument("--model_name", type=str, default="lenet5")
    parser.add_argument("--conv_type", type=int, default=1)
    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--signed", action="store_true", default=False)
    parser.add_argument("--zone", action="store_true", default=False)
    parser.add_argument("--multiplier_matrix", type=str, default=None)
    args = parser.parse_args()

    model_name = normalize_model_name(args.model_name)
    models_dir = trained_models_path.rstrip("/")
    features_dir = os.path.join(CURR_DIR, "epoch_features")     # this tool's raw-activation output
    artifact_dir = os.path.join(CURR_DIR, "epoch_artifacts")   # this tool's embedded-coordinate output

    cnn_training.set_data_loaders(model_name)
    num_classes = cnn_training._classes if cnn_training._classes else 10
    # empty untrained model shell - reused every iteration, with only its weights changing epoch in epoch
    model = build_model(model_name, conv_type=args.conv_type, bit_width=args.bit_width, signed=args.signed, 
                        zone=args.zone, multiplier_matrix=args.multiplier_matrix, num_classes=num_classes)

    layer_path = resolve_tsne_layer_path(model, "penultimate")
    exact_epochs, _, _ = get_exact_training_setup(model_name, model)
    stage_tag, ckpt_template, epochs = resolve_stage(args.conv_type, args.bit_width,
                                                                args.multiplier_matrix, exact_epochs)

    prev_embedding = None
    for epoch_num in range(1, epochs + 1):
        ckpt_path = os.path.join(models_dir, ckpt_template.format(model_name=model_name, epoch=epoch_num))
        model.load_state_dict(torch.load(ckpt_path))
        if args.conv_type != 1:
            calibration(model)
            model.eval()

        X_feat, y, y_pred = _collect_layer_features(model, cnn_training.test_loader, device, layer_path, collect_preds=True)
        save_epoch_features(features_dir, stage_tag, epoch_num, X_feat, y, y_pred)
        
        X_3d = compute_epoch_embedding(X_feat, prev_embedding=prev_embedding)
        save_epoch_artifact(artifact_dir, stage_tag, epoch_num, X_3d, y, y_pred)

        prev_embedding = X_3d
        
    save_train_manifest(
        os.path.join(CURR_DIR, "dashboard_data", "train_manifest.json"),
        features_dir,
        artifact_dir
    )

if __name__ == "__main__":
    main()