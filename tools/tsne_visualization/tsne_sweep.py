from __future__ import annotations  # compiler directive

import numpy as np
from sklearn.manifold import TSNE
from statistics import median

def run_tsne_sweep(train_schedule: list[int],
                   test_features: nd.ndarray,    # fixed test features, shape (test_size, D)
                   feature_pool: np.ndarray,    # extracted features for full train pool, shape (max_train, D)
                   seed:int,
                   components: str,
                   perplexity: float = 30,
                   max_iter: int = 1000,
                   n_repeats: int = 5):

    """Nested t-SNE sweep: for each N in train_schedule, take the first N rows of feature_pool, 
    combine with test_features, fit a fresh TSNE, and record (N, kl_divergence, coords) for that step."""

    # derive n_repeats seeds from one starting seed (shared across steps)
    # to maintain reproducible results
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_repeats)
    seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

    results = []

    for n_train in train_schedule:
        prefix = feature_pool[:n_train]
        X_step = np.concatenate([prefix, test_features])
        n_total = n_train + len(test_features)

        if perplexity >= n_total:
            raise ValueError(f"Perplexity ({perplexity}) must be less than n_total ({n_total})")

        # check what dimensional reductions to compute and run TSNE
        compute_2d = components in ("2D", "2D+3D")
        compute_3d = components in ("3D", "2D+3D")

        X_2d_repeats, kl_div_2d = [], []
        X_3d_repeats, kl_div_3d = [], []

        for s in seeds:
            if compute_2d:
                tsne_2d = TSNE(n_components=2,
                            perplexity=perplexity,
                            n_iter=max_iter,
                            init="pca",
                            random_state=s)
                X_2d_repeats.append(tsne_2d.fit_transform(X_step))
                kl_div_2d.append(tsne_2d.kl_divergence_)

            if compute_3d:
                tsne_3d = TSNE(n_components=3,
                            perplexity=perplexity,
                            n_iter=max_iter,
                            init="pca",
                            random_state=s)
                X_3d_repeats.append(tsne_3d.fit_transform(X_step))
                kl_div_3d.append(tsne_3d.kl_divergence_)

        median_2d = median(kl_div_2d) if compute_2d else None
        median_3d = median(kl_div_3d) if compute_3d else None

        # find repeat that is jointly closest to median
        if compute_2d and compute_3d:
            best_i = min(
                range(len(kl_div_2d)),
                key=lambda i: abs(kl_div_2d[i] - median_2d) / median_2d
                            + abs(kl_div_3d[i] - median_3d) / median_3d
            )
            X_2d, X_3d = X_2d_repeats[best_i], X_3d_repeats[best_i]
            
        elif compute_2d:
            best_i = min(range(len(kl_div_2d)), key=lambda i: abs(kl_div_2d[i] - median_2d))
            X_2d, X_3d = X_2d_repeats[best_i], None
        elif compute_3d:
            best_i = min(range(len(kl_div_3d)), key=lambda i: abs(kl_div_3d[i] - median_3d))
            X_2d, X_3d = None, X_3d_repeats[best_i]
        else:
            X_2d, X_3d = None, None

        metadata = {"n_train": n_train,
                    "n_total": n_total,
                    "X_2d": X_2d,
                    "X_3d": X_3d,
                    "kl_divergence_2d": kl_div_2d,
                    "kl_divergence_3d": kl_div_3d,
                    "median_2d": median_2d,
                    "median_3d": median_3d}
        results.append(metadata)
    
    return results