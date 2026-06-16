import os
from pathlib import Path

import numpy as np


def safe_layer_name(layer_path):
    """Convert a layer path string into a safe filesystem string."""
    if not layer_path:
        return ""
    return (str(layer_path)
            .replace(".", "-")
            .replace("/", "-")
            .replace("[", "")
            .replace("]", ""))


def build_layer_tag(feature_space, feature_layer_path):
    """Construct the layer portion of an artifact filename."""
    if feature_space == "layer" and feature_layer_path:
        return f"_layer-{safe_layer_name(feature_layer_path)}"
    return ""


def build_classes_tag(classes):
    """Construct the classes portion of an artifact filename."""
    if classes is None:
        return ""
    return "_classes" + "-".join(str(c) for c in classes)


def build_tag(output_tag):
    """Construct the user-provided tag portion of an artifact filename."""
    return f"_{output_tag}" if output_tag else ""


def build_dash_data_dir(save_dir, feature_space, model_name, run_id=None):
    """Construct the directory path for storing Dash .npz artifacts."""
    if run_id:
        return os.path.join(save_dir, feature_space, model_name, run_id, "dash_data")
    return os.path.join(save_dir, feature_space, model_name, "dash_data")


def build_dash_artifact_basename(model_name, feature_space, feature_layer_path,
                                 output_tag, classes):
    """Construct the base filename (without extension) for a Dash artifact."""
    layer_tag = build_layer_tag(feature_space, feature_layer_path)
    tag = build_tag(output_tag)
    classes_tag = build_classes_tag(classes)
    return f"tsne_{model_name}{layer_tag}{tag}{classes_tag}"


def build_dash_artifact_path(save_dir, feature_space, model_name, feature_layer_path,
                             output_tag, classes, run_id=None):
    """Construct the full canonical .npz path for a given run configuration.
    
    The path follows the layout:
        <save_dir>/<feature_space>/<model_name>/[run_id]/dash_data/<basename>.npz
    where <basename> encodes layer, tag, and class filter for uniqueness.
    """
    dash_out_dir = build_dash_data_dir(save_dir, feature_space, model_name, run_id)
    basename = build_dash_artifact_basename(
        model_name=model_name,
        feature_space=feature_space,
        feature_layer_path=feature_layer_path,
        output_tag=output_tag,
        classes=classes,
    )
    return os.path.join(dash_out_dir, f"{basename}.npz")


def build_dash_artifact_pattern(save_dir, feature_space, model_name, feature_layer_path,
                                output_tag, run_id=None):
    """Construct a glob pattern to find artifacts matching a run configuration."""
    dash_out_dir = build_dash_data_dir(save_dir, feature_space, model_name, run_id)
    layer_tag = build_layer_tag(feature_space, feature_layer_path)
    tag = build_tag(output_tag)
    return os.path.join(dash_out_dir, f"tsne_{model_name}{layer_tag}{tag}*.npz")


def get_misclassified_indices(y_test_sub, y_pred_sub):
    """Return indices of incorrect predictions within a test subset."""
    y_test_sub = np.asarray(y_test_sub)
    y_pred_sub = np.asarray(y_pred_sub)
    return np.where(y_test_sub != y_pred_sub)[0]


def save_dash_artifact(artifact_path, X_2d, y_all, test_mask, y_test_sub,
                       y_pred_sub, X_test_pixels, image_shape,
                       model_name, feature_space, feature_layer_path,
                       feature_layer_requested, output_tag, classes, title):
    """Persist t-SNE plot data and metadata into a compressed .npz archive."""
    os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
    np.savez_compressed(
        artifact_path,
        X_2d=np.asarray(X_2d),
        y_all=np.asarray(y_all),
        test_mask=np.asarray(test_mask, dtype=bool),
        y_test_sub=np.asarray(y_test_sub),
        y_pred_sub=np.asarray(y_pred_sub),
        X_test_pixels=np.asarray(X_test_pixels),
        image_shape=np.asarray(image_shape if image_shape is not None else [], dtype=np.int64),
        model_name=np.array(model_name),
        feature_space=np.array(feature_space),
        feature_layer_path=np.array(feature_layer_path if feature_layer_path is not None else ""),
        feature_layer_requested=np.array(
            feature_layer_requested if feature_layer_requested is not None else ""
        ),
        output_tag=np.array(output_tag if output_tag is not None else ""),
        classes=np.asarray(classes if classes is not None else [], dtype=np.int64),
        title=np.array(title),
    )


def load_dash_artifact(path):
    """Load a .npz Dash artifact and return its contents as a Python dictionary.
    
    Keys: X_2d, y_all, test_mask, y_test_sub, y_pred_sub, X_test_pixels,
          image_shape, model_name, feature_space, feature_layer_path,
          feature_layer_requested, output_tag, title.
    All numpy scalars are already cast to plain Python types.
    """
    data = np.load(path, allow_pickle=False)
    return {
        "path": str(path),
        "X_2d": data["X_2d"],
        "y_all": data["y_all"],
        "test_mask": data["test_mask"].astype(bool),
        "y_test_sub": data["y_test_sub"],
        "y_pred_sub": data["y_pred_sub"],
        "X_test_pixels": data["X_test_pixels"],
        "image_shape": tuple(int(v) for v in data["image_shape"].tolist()) if "image_shape" in data else (),
        "model_name": str(data["model_name"].item()) if "model_name" in data else "",
        "feature_space": str(data["feature_space"].item()) if "feature_space" in data else "",
        "feature_layer_path": str(data["feature_layer_path"].item()) if "feature_layer_path" in data else "",
        "feature_layer_requested": str(data["feature_layer_requested"].item()) if "feature_layer_requested" in data else "",
        "output_tag": str(data["output_tag"].item()) if "output_tag" in data else "",
        "title": str(data["title"].item()) if "title" in data else "t-SNE",
    }


def latest_artifact_after(save_dir, feature_space, model_name, start_ts):
    """Find the most recently modified artifact created after a given timestamp."""
    base_dir = Path(save_dir) / feature_space / model_name
    candidates = list(base_dir.rglob("dash_data/*.npz"))
    candidates = [p for p in candidates if p.stat().st_mtime >= start_ts]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])
