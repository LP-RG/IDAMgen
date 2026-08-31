import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import torch
import tsne_sweep
import tsne_sweep_figures
from sklearn.manifold import TSNE
from tsne_utils import (
    build_classes_tag,
    build_dash_artifact_path,
    build_dash_artifact_basename,
    build_layer_tag,
    build_tag,
    save_dash_artifact,
    save_sweep_step_artifact,
    save_sweep_manifest
)


def _filter_loader(loader, classes):
    """Return a new DataLoader restricted to the given class labels."""
    targets = np.array(loader.dataset.targets)
    keep_idx = np.where(np.isin(targets, classes))[0]
    subset = torch.utils.data.Subset(loader.dataset, keep_idx)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
    )


def _subsample_loader(loader, n_max, rng, indices=None):
    """Return a new DataLoader with a deterministic or random sample subset."""
    n_total = len(loader.dataset)
    if indices is None:
        n = min(n_max, n_total)
        idx = rng.choice(n_total, n, replace=False)
    else:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("indices must be a 1D array-like of sample indices.")
        if len(idx) == 0:
            raise ValueError("indices cannot be empty.")
        if idx.min() < 0 or idx.max() >= n_total:
            raise ValueError("Provided indices are out of bounds for loader.dataset.")
    subset = torch.utils.data.Subset(loader.dataset, idx)
    sub_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
    )
    return sub_loader, idx


def _collect_inputs(loader):
    """Collect raw inputs and labels from a DataLoader without running the model."""
    inputs_list, labels_list = [], []
    for inputs, labels in loader:
        inputs_list.append(inputs.cpu())
        labels_list.append(labels.cpu())
    X = torch.cat(inputs_list).numpy()
    y = torch.cat(labels_list).numpy()
    return X.reshape(len(X), -1), y


def _collect_predictions(model, loader, device):
    """Forward-pass over a DataLoader; return (X_flat, y_true, y_pred)."""
    model.eval()
    inputs_list, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            inputs_list.append(inputs.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())

    X = torch.cat(inputs_list).numpy()
    y = torch.cat(labels_list).numpy()
    y_pred = torch.cat(preds_list).numpy()
    return X.reshape(len(X), -1), y, y_pred


def _collect_layer_features(model, loader, device, layer_path, collect_preds=False):
    """Collect activations from an arbitrary module path via forward hook."""
    modules = dict(model.named_modules())
    if layer_path not in modules:
        available = ", ".join(sorted(modules.keys()))
        raise ValueError(
            f"Layer '{layer_path}' not found in model. Available modules: {available}"
        )

    target_layer = modules[layer_path]
    model.eval()
    features_list, labels_list, preds_list = [], [], []

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        feat = output.detach().cpu()
        feat = feat.reshape(feat.shape[0], -1)
        features_list.append(feat)

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                labels_list.append(labels.cpu())
                if collect_preds:
                    _, preds = outputs.max(1)
                    preds_list.append(preds.cpu())
    finally:
        handle.remove()

    X_feat = torch.cat(features_list).numpy()
    y = torch.cat(labels_list).numpy()
    if collect_preds:
        y_pred = torch.cat(preds_list).numpy()
        return X_feat, y, y_pred
    return X_feat, y


def plot_tsne_embedding_cnn(X_2d, y, title=None, y_pred=None, test_mask=None,
                            save_path=None):
    """Plots t-SNE embedding with CNN misclassification overlay."""
    X_2d = np.asarray(X_2d)
    y = np.asarray(y)

    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    palette = np.array(sns.color_palette("hls", n_classes))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    color_indices = np.array([label_to_idx[int(lab)] for lab in y])

    f, ax = plt.subplots(figsize=(10, 8))

    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool)
        train_mask = ~test_mask

        ax.scatter(X_2d[train_mask, 0], X_2d[train_mask, 1],
                   lw=0, s=15, c="#cccccc", alpha=0.4, zorder=1,
                   label="Train (not evaluated)")
        ax.scatter(X_2d[test_mask, 0], X_2d[test_mask, 1],
                   lw=0, s=50, c=palette[color_indices[test_mask]],
                   alpha=0.85, zorder=2, label="Test (true label)")
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   lw=0, s=40, c=palette[color_indices],
                   alpha=0.7, zorder=2)

    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        eval_mask = test_mask if test_mask is not None else np.ones(len(y), dtype=bool)
        y_eval = y[eval_mask]

        wrong_in_eval = y_eval != y_pred
        wrong = np.zeros(len(y), dtype=bool)
        wrong[eval_mask] = wrong_in_eval

        if wrong.sum() > 0:
            ax.scatter(X_2d[wrong, 0], X_2d[wrong, 1],
                       marker="x", s=120, c="red", linewidths=1.8,
                       label=f"Misclassified ({wrong.sum()})", zorder=5)

        acc = (y_eval == y_pred).mean()
        ax.annotate(
            f"CNN accuracy: {acc:.1%}",
            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.axis("off")
    ax.axis("tight")

    ref_mask = (test_mask if test_mask is not None
                else np.ones(len(y), dtype=bool))
    for lab in unique_labels:
        pts = X_2d[ref_mask & (y == lab)]
        if len(pts) == 0:
            continue
        xtext, ytext = np.median(pts, axis=0)
        txt = ax.text(xtext, ytext, str(lab), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal(),
        ])

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        f.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close(f)
    return f, ax


def show_misclassified_images(X, y, y_pred, image_shape, save_path,
                              classes=None, n_cols=8):
    """Display a grid of CNN-misclassified test images."""
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    wrong = y != y_pred
    if classes is not None:
        classes = np.asarray(classes)
        wrong = wrong & np.isin(y, classes)

    indices = np.where(wrong)[0]
    if len(indices) == 0:
        print("No misclassified images to show.")
        return

    n = len(indices)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    c, h, w = image_shape

    classes_str = f" (classes {classes.tolist()})" if classes is not None else ""
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.4, n_rows * 1.8 + 0.4))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(indices):
        img = X[idx].reshape(c, h, w)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if c == 1:
            axes[ax_i].imshow(img[0], cmap="gray_r", interpolation="nearest")
        else:
            axes[ax_i].imshow(img.transpose(1, 2, 0), interpolation="nearest")
        axes[ax_i].set_title(f"#{idx}\ntrue={y[idx]}\npred={y_pred[idx]}",
                             fontsize=7, color="red")
        axes[ax_i].axis("off")

    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(f"CNN misclassifications{classes_str} — {n} samples", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Misclassified images saved to {save_path}")
    plt.close(fig)


def run_tsne_cnn_experiment(model, train_loader, test_loader, device,
                            model_name="lenet5", perplexity=30, components="2D",
                            max_iter=1000, max_train=2000, max_test=1000,
                            save_dir=None, classes=None, seed=42,
                            show_misclassifications=False, image_shape=None,
                            feature_space="layer", output_tag=None,
                            feature_layer_path=None,
                            feature_layer_requested=None,
                            save_static=True,
                            save_dash_artifact=True,
                            subsample_state=None,
                            run_id=None):
    
    # ------------------------------------------------------------------ #
    # CORREZIONE: Se save_dir non viene passato dal main o è relativo, 
    # lo forziamo ad essere un percorso assoluto rispetto alla cartella di questo file.
    # ------------------------------------------------------------------ #
    if save_dir is None or save_dir == "./plots":
        # Prende la cartella in cui si trova QUESTO file (tsne_visualization.py)
        # Se preferisci che si agganci al main, il main ora gli passerà CURR_DIR/plots,
        # che è un path assoluto e salterà questo if.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "plots")
    else:
        # Se il main passa un percorso, ci assicuriamo che sia assoluto
        save_dir = os.path.abspath(save_dir)

    rng = np.random.default_rng(seed)

    if classes is not None:
        classes = np.asarray(classes)
        train_loader = _filter_loader(train_loader, classes)
        test_loader = _filter_loader(test_loader, classes)
        print(f"Filtered loaders to classes {classes.tolist()}: "
              f"{len(train_loader.dataset)} train, "
              f"{len(test_loader.dataset)} test samples")

    train_indices = None
    test_indices = None
    if subsample_state is not None:
        train_indices = subsample_state.get("train_indices")
        test_indices = subsample_state.get("test_indices")

    train_loader, train_indices = _subsample_loader(
        train_loader, max_train, rng, indices=train_indices
    )
    test_loader, test_indices = _subsample_loader(
        test_loader, max_test, rng, indices=test_indices
    )
    if subsample_state is not None:
        subsample_state["train_indices"] = train_indices
        subsample_state["test_indices"] = test_indices

    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)

    if feature_space == "layer":
        if feature_layer_path is None:
            raise ValueError("feature_layer_path is required when feature_space='layer'")
        print(f"Collecting training features from layer '{feature_layer_path}'...")
        X_train_sub, y_train_sub = _collect_layer_features(
            model, train_loader, device, layer_path=feature_layer_path
        )
    elif feature_space == "pixels":
        print("Collecting training inputs...")
        X_train_sub, y_train_sub = _collect_inputs(train_loader)
    else:
        raise ValueError("feature_space must be one of: 'layer', 'pixels'")

    print("Collecting CNN predictions on test set...")
    if feature_space == "layer":
        X_test_sub, y_test_sub, y_pred_sub = _collect_layer_features(
            model, test_loader, device, layer_path=feature_layer_path, collect_preds=True
        )
        X_test_pixels, y_test_pixels = _collect_inputs(test_loader)
        if not np.array_equal(y_test_pixels, y_test_sub):
            raise RuntimeError(
                "Mismatch between pixel-label order and layer-label order while "
                "preparing misclassification image grid / dash artifacts."
            )
    else:
        X_test_sub, y_test_sub, y_pred_sub = _collect_predictions(model, test_loader, device)
        X_test_pixels = X_test_sub

    X_all = np.concatenate([X_train_sub, X_test_sub])
    y_all = np.concatenate([y_train_sub, y_test_sub])

    test_mask = np.zeros(len(y_all), dtype=bool)
    test_mask[n_train:] = True

    n_features = X_all.shape[1]
    print(f"Running t-SNE on {len(X_all)} samples "
          f"({n_train} train + {n_test} test), {n_features} features...")
    
    # check what dimensional reductions to compute and run TSNE
    compute_2d = components in ("2D", "2D+3D")
    compute_3d = components in ("3D", "2D+3D")

    X_2d = None
    if compute_2d:
        X_2d = TSNE(n_components=2,
        perplexity=perplexity,
        n_iter=max_iter,
        init="pca",
        random_state=seed,
    ).fit_transform(X_all)

    X_3d = None
    if compute_3d:
        X_3d = TSNE(n_components=3,
        perplexity=perplexity,
        n_iter=max_iter,
        init="pca",
        random_state=seed,
    ).fit_transform(X_all)

    wrong_count = int((y_test_sub != y_pred_sub).sum())
    acc = (y_test_sub == y_pred_sub).mean()
    print(f"CNN test accuracy (subsample): {acc:.1%} | "
          f"misclassified: {wrong_count}/{n_test}")

    classes_tag = build_classes_tag(classes)
    
    # Ora tsne_out_dir e mis_out_dir useranno il save_dir ASSOLUTO
    if run_id:
        tsne_out_dir = os.path.join(save_dir, feature_space, model_name, run_id, "tsne")
        mis_out_dir = os.path.join(save_dir, feature_space, model_name, run_id, "misclassified")
    else:
        tsne_out_dir = os.path.join(save_dir, feature_space, model_name, "tsne")
        mis_out_dir = os.path.join(save_dir, feature_space, model_name, "misclassified")
        
    tag = build_tag(output_tag)
    layer_tag = build_layer_tag(feature_space, feature_layer_path)

    classes_str = (f"classes {classes.tolist()}"
                   if classes is not None else "all classes")
    layer_str = feature_layer_path if feature_space == "layer" else "raw-pixels"
    run_tag_str = output_tag if output_tag is not None else "default"
    title = (f"t-SNE of {model_name} {feature_space} features ({classes_str})\n"
             f"run: {run_tag_str} | layer: {layer_str}\n"
             f"red x = CNN misclassification | acc: {acc:.1%}")
             
    if save_static and X_2d is not None:
        save_path = os.path.join(
            tsne_out_dir,
            f"tsne_{model_name}{layer_tag}{tag}{classes_tag}.png",
        )
        plot_tsne_embedding_cnn(
            X_2d, y_all,
            title=title,
            y_pred=y_pred_sub,
            test_mask=test_mask,
            save_path=save_path,
        )
    elif save_static and X_2d is None:
        print("Skipping static PNG export  - no 2D coordinates computed for this run (components = '3D')")
    else:
        print("Skipping static t-SNE PNG export (--tsne-no-save-static).")

    if save_dash_artifact:
        # Passiamo il save_dir forzato ad essere assoluto
        artifact_path = build_dash_artifact_path(
            save_dir=save_dir,
            feature_space=feature_space,
            model_name=model_name,
            feature_layer_path=feature_layer_path,
            output_tag=output_tag,
            classes=classes,
            run_id=run_id
        )
        save_dash_artifact(
            artifact_path,
            X_2d=X_2d,
            X_3d=X_3d,
            y_all=y_all,
            test_mask=test_mask,
            y_test_sub=y_test_sub,
            y_pred_sub=y_pred_sub,
            X_test_pixels=X_test_pixels,
            image_shape=image_shape,
            model_name=model_name,
            feature_space=feature_space,
            feature_layer_path=feature_layer_path,
            feature_layer_requested=feature_layer_requested,
            output_tag=output_tag,
            classes=classes,
            title=title
        )
        print(f"Dash artifact saved to {artifact_path}")
    else:
        print("Skipping Dash artifact export (--tsne-no-save-dash-artifact).")

    if show_misclassifications:
        if not save_static:
            print("Skipping misclassification grid because static export is disabled.")
            return X_2d, X_3d, y_all, y_pred_sub, test_mask
        if image_shape is None:
            raise ValueError("image_shape=(C, H, W) is required when show_misclassifications=True")
        errors_path = os.path.join(
            mis_out_dir,
            f"misclassified_{model_name}{layer_tag}{tag}{classes_tag}.png",
        )
        show_misclassified_images(X_test_pixels, y_test_sub, y_pred_sub,
                                  image_shape=image_shape, classes=classes,
                                  save_path=errors_path)

    return X_2d, X_3d, y_all, y_pred_sub, test_mask


def run_tsne_sweep_cnn_experiment(model, train_loader, test_loader, device, train_schedule,
                            model_name="lenet5", perplexity=30, components="2D",
                            max_iter=1000, test_size=1000,
                            save_dir=None, seed=42, image_shape=None,
                            feature_layer_path=None,
                            feature_layer_requested=None,
                            save_artifact=True, 
                            run_id=None, run_dir=None, n_repeats=5):

    """Run the full N_total sweep: extract features once, sweep t-SNE over train_schedule,
    save per-step artifacts, save the sweep manifest, and build the summary curve figure."""

    rng = np.random.default_rng(seed)
    N_max = max(train_schedule)

    train_loader, train_indices = _subsample_loader(train_loader, N_max, rng)
    n_pool = len(train_loader.dataset)
    test_loader, test_indices = _subsample_loader(test_loader, test_size, rng)
    
    feature_pool, y_train_pool = _collect_layer_features(
        model, train_loader, device, layer_path=feature_layer_path
    )
    
    test_features, y_test_sub, y_pred_sub = _collect_layer_features(
        model, test_loader, device, layer_path=feature_layer_path, collect_preds=True
    )

    X_test_pixels, y_test_pixels = _collect_inputs(test_loader)
    if not np.array_equal(y_test_pixels, y_test_sub):
        raise RuntimeError(
            "Mismatch between pixel-label order and layer-label order while "
            "preparing sweep dash artifacts."
        )

    step_results = tsne_sweep.run_tsne_sweep(
        train_schedule, test_features, feature_pool, 
        seed, components, perplexity, max_iter, n_repeats
    )

    manifest_rows = []
    for step in step_results:
        n_train_step = step["n_train"]
        n_total_step = step["n_total"]
        
        y_all = np.concatenate([y_train_pool[:n_train_step], y_test_sub])
        test_mask = np.zeros(n_total_step, dtype=bool)
        test_mask[n_train_step:] = True
        
        output_tag = f"sweep_n{n_total_step}"
        artifact_path = None

        if save_artifact:
            basename = build_dash_artifact_basename(
                model_name=model_name, feature_space="layer",
                feature_layer_path=feature_layer_path, output_tag=output_tag, classes=None
            )
            artifact_path = os.path.join(save_dir, "layer", model_name, "sweep", run_id, "dash_data", f"{basename}.npz")
            save_sweep_step_artifact(
                sweep_path=artifact_path,
                n_train=n_train_step,
                n_total=n_total_step,
                X_2d=step.get("X_2d"),
                X_3d=step.get("X_3d"),
                kl_div_2d=step.get("kl_divergence_2d"),
                kl_div_3d=step.get("kl_divergence_3d"),
                median_2d=step.get("median_2d"),
                median_3d=step.get("median_3d"),
                y_all=y_all,
                test_mask=test_mask
            )
            print(f"Dash artifact saved to {artifact_path}")
        else:
            print("Skipping Dash artifact export (--tsne-no-save-dash-artifact).")

        manifest_rows.append({
            "n_train": n_train_step,
            "n_total": n_total_step,
            "kl_divergence_2d": step.get("kl_divergence_2d"),
            "kl_divergence_3d": step.get("kl_divergence_3d"),
            "median_2d": step.get("median_2d"),
            "median_3d": step.get("median_3d"),
            "y_all": y_all,
            "test_mask": test_mask,
            "artifact_path": artifact_path,
        })

    fig = tsne_sweep_figures.build_kld_sweep_figure(manifest_rows)

    curve_path_html = os.path.join(run_dir, "kld_sweep_curve.html")
    fig.write_html(curve_path_html)
    
    curve_path_png = os.path.join(run_dir, "kld_sweep_curve.png")
    try:
        fig.write_image(curve_path_png)
    except Exception as e:
        print(f"Skipping PNG export: {e}")

    save_sweep_manifest(run_dir, manifest_rows)
    return run_dir, manifest_rows