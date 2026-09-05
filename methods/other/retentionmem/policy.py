"""Small two-hidden-layer MLP, trained with NumPy; inference needs no PyTorch."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -40, 40)))


class RetentionPolicy:
    mode = "learned"

    def __init__(self, parameters, mean, scale, metadata):
        self.p = parameters
        self.mean, self.scale = mean, scale
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])

    def predict(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Policy feature shape/value mismatch")
        z = (x-self.mean)/self.scale
        h1 = np.maximum(z @ self.p["w1"] + self.p["b1"], 0)
        h2 = np.maximum(h1 @ self.p["w2"] + self.p["b2"], 0)
        return sigmoid(h2 @ self.p["w3"] + self.p["b3"]).reshape(-1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # No pickle: checkpoints contain only arrays and a UTF-8 JSON scalar.
        with path.open("wb") as stream:
            np.savez_compressed(stream, **self.p, mean=self.mean, scale=self.scale,
                                metadata=np.asarray(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path, fingerprint=None, allow_test_artifact=False):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Learned policy missing: {path}. Run collect and train first; use --mode heuristic only for the named control.")
        with np.load(path, allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            if fingerprint is not None and metadata["features"] != fingerprint:
                raise ValueError("Policy embedding/tokenizer/feature configuration does not match this run")
            if metadata.get("test_artifact") and not allow_test_artifact:
                raise ValueError("A test/demo checkpoint cannot be used for scientific evaluation")
            return cls({k: value[k].copy() for k in ("w1", "b1", "w2", "b2", "w3", "b3")},
                       value["mean"].copy(), value["scale"].copy(), metadata)


def calibrate(scores, labels, candidate_ids, max_regression_rate=0.05, min_accepted=20):
    """Empirical held-out calibration, not a statistical safety guarantee."""
    if not len(scores) == len(labels) == len(candidate_ids) or not 0 <= max_regression_rate <= 1 or min_accepted < 1:
        raise ValueError("Invalid calibration labels, candidate IDs or acceptance settings")
    if not np.isfinite(scores).all() or not set(np.asarray(labels).ravel().tolist()) <= {0, 1}:
        raise ValueError("Invalid calibration scores/labels")
    groups = {}
    for s, y, group in zip(scores, labels, candidate_ids):
        old = groups.get(str(group), (0., 0))
        groups[str(group)] = (max(old[0], float(s)), max(old[1], int(y)))
    if not groups:
        raise ValueError("Calibration requires validation candidates")
    pairs = np.asarray(list(groups.values()))
    threshold, count, rate = -1.0, 0, None
    for tau in sorted(set(pairs[:, 0].tolist())):
        accepted = pairs[pairs[:, 0] <= tau]
        current_rate = float(accepted[:, 1].mean())
        if len(accepted) >= min_accepted and current_rate <= max_regression_rate:
            threshold, count, rate = float(tau), len(accepted), current_rate
    return threshold, {"accepted_candidates": count, "validation_candidates": len(groups),
                       "empirical_regression_rate": rate, "max_regression_rate": max_regression_rate,
                       "min_accepted": min_accepted, "guarantee": False}


def train_policy(x, y, validation_x, validation_y, validation_ids, *, fingerprint, seed=0,
                 epochs=80, batch_size=128, learning_rate=0.001, max_regression_rate=0.05,
                 min_accepted=20, test_artifact=False):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    if x.ndim != 2 or validation_x.ndim != 2 or x.shape[1] != validation_x.shape[1] or len(x) != len(y):
        raise ValueError("Training/validation feature and label shapes do not match")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Training hyperparameters must be positive")
    if len(x) < 2 or set(y.ravel().tolist()) != {0., 1.}:
        raise ValueError("Training requires both regressing and non-regressing examples")
    if not np.isfinite(x).all() or not np.isfinite(validation_x).all():
        raise ValueError("Nonfinite training features")
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-3)
    z = (x-mean)/scale
    dims = (x.shape[1], 256, 64, 1)
    p = {}
    for i in range(1, 4):
        p[f"w{i}"] = (rng.standard_normal((dims[i-1], dims[i])) * np.sqrt(2/dims[i-1])).astype(np.float32)
        p[f"b{i}"] = np.zeros(dims[i], np.float32)
    m, v = {k: np.zeros_like(a) for k, a in p.items()}, {k: np.zeros_like(a) for k, a in p.items()}
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(z))
        for begin in range(0, len(order), batch_size):
            ids = order[begin:begin+batch_size]
            a = z[ids]
            h1 = np.maximum(a @ p["w1"] + p["b1"], 0)
            h2 = np.maximum(h1 @ p["w2"] + p["b2"], 0)
            d3 = (sigmoid(h2 @ p["w3"] + p["b3"])-y[ids])/len(ids)
            d2 = (d3 @ p["w3"].T)*(h2 > 0)
            d1 = (d2 @ p["w2"].T)*(h1 > 0)
            g = {"w3": h2.T @ d3, "b3": d3.sum(axis=0), "w2": h1.T @ d2,
                 "b2": d2.sum(axis=0), "w1": a.T @ d1, "b1": d1.sum(axis=0)}
            step += 1
            for k in p:
                grad = g[k] + (1e-5*p[k] if k.startswith("w") else 0)
                m[k] = .9*m[k]+.1*grad
                v[k] = .999*v[k]+.001*grad*grad
                p[k] -= learning_rate*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    metadata = {"seed": seed, "features": fingerprint, "threshold": .5, "epochs": epochs,
                "training_examples": len(x), "test_artifact": test_artifact,
                "architecture": [dims[0], 256, 64, 1]}
    model = RetentionPolicy(p, mean, scale, metadata)
    threshold, report = calibrate(model.predict(validation_x), validation_y, validation_ids,
                                  max_regression_rate, min_accepted)
    model.threshold = threshold
    metadata.update(threshold=threshold, calibration=report)
    return model
