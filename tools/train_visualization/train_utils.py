import os
import numpy as np
import json
import glob
import re

MANIFEST_ARTIFACT_PATTERN = re.compile(r"^(?P<stage_tag>.+)_epoch(?P<epoch_num>\d+)_artifact\.npz$")

def save_epoch_features(save_dir, stage_tag, epoch_num, X_feat, y, y_pred):
    """Write one epoch's raw activations to disk.
    save_dir: directory to write into
    stage_tag: records which run this came from and keeps from overwriting
    epoch_num: epoch the snapshot came from
    X_feat: ndarray with shape (n_test, feature_dim) - raw activations from collect_feature_layers
    y: ndarray, shape (n_test, ) - true labels
    y_pred: ndarray, shape (n_test, )  - predicted labels
    """
    os.makedirs(save_dir, exist_ok=True)    # recursive directory creation function
    path = os.path.join(save_dir, f"{stage_tag}_epoch{epoch_num}_features.npz")
    np.savez_compressed(
        path,
        epoch=np.array(epoch_num),
        X_feat=np.asarray(X_feat),
        y=np.asarray(y),
        y_pred=np.asarray(y_pred)
    )
    return path


def load_epoch_features(path):
    """Load one epoch's raw activations back into a plain dict."""
    data = np.load(path, allow_pickle=False)
    return {
        "epoch": int(data["epoch"]),
        "X_feat": data["X_feat"],
        "y": data["y"],
        "y_pred": data["y_pred"]
    }    


def save_epoch_artifact(save_dir, stage_tag, epoch_num, X_3d, y, y_pred):
    """Write one epoch's 3D embedding and labels to disk storage."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{stage_tag}_epoch{epoch_num}_artifact.npz")
    np.savez_compressed(path, epoch=np.array(epoch_num), X_3d=np.asarray(X_3d), 
                        y=np.asarray(y), y_pred=np.asarray(y_pred))
    return path


def load_epoch_artifact(path):
    """Load one epoch's saved embadding back into the a plain dict."""
    data = np.load(path, allow_pickle=False)
    return {"epoch": int(data["epoch"]), "X_3d": data["X_3d"],
            "y": data["y"], "y_pred": data["y_pred"]}


def save_train_manifest(manifest_path, features_dir, artifact_dir):
    """Scan artifacts_dir for every {stage_tag}_epoch{N}_artifact.npz file and write
    one combined manifest mapping each stage/epoch to its artifact and features paths."""
    epochs = []
    for artifact_path in glob.glob(os.path.join(artifact_dir, "*_artifact.npz")):
        match = MANIFEST_ARTIFACT_PATTERN.match(os.path.basename(artifact_path))
        if not match:
            continue
        stage_tag = match.group("stage_tag")
        epoch_num = int(match.group("epoch_num"))
        features_path = os.path.join(features_dir, f"{stage_tag}_epoch{epoch_num}_features.npz")

        epochs.append({
            "stage_tag": stage_tag,
            "epoch": epoch_num,
            "artifact_path": artifact_path,
            "features_path": features_path
        })
    # glob.glob returns full paths in arbitrary order - filenames extracted with basename
    epochs.sort(key=lambda e: (e["stage_tag"], e["epoch"]))
    manifest = {"epochs": epochs}
    
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_train_manifest(manifest_path):
    """Load the combined stage, epoch manifest JSON."""
    with open(manifest_path) as f:
        return json.load(f)


def get_artifact_path(manifest, stage_tag, epoch_num):
    """Look up one (stage_tag, epoch) pair's artifact path in the manifest."""
    for entry in manifest["epochs"]:
        if entry["stage_tag"] == stage_tag and entry["epoch"] == epoch_num:
            return entry["artifact_path"]
    raise ValueError(f"No artifact for stage_tag={stage_tag}, epoch={epoch_num}")


def get_max_epoch(manifest, stage_tag):
    """Return the highest epoch number recorded for a given stage."""
    m = 0
    for entry in manifest["epochs"]:
        if entry["stage_tag"] == stage_tag and entry["epoch"] > m:
            m = entry["epoch"]
    if m == 0:
        raise ValueError(f"No manifest epochs' entry for stage_tag={stage_tag}")
    return m