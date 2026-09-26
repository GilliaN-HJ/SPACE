from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .slow_basis import SlowPredictiveBasis
from .utility import UtilityBank


def load_slow_basis(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> SlowPredictiveBasis:
    """Load a train-only slow predictive basis artifact."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "basis",
        "eigenvalues",
        "short_lag",
        "long_lag",
        "regularization",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"basis artifact is missing fields: {sorted(missing)}")
    if payload.get("fit_split", "train") != "train":
        raise ValueError("SPACE bases must be fitted on the training split")
    if payload.get("validation_features_accessed", False) is not False:
        raise ValueError("basis artifact reports validation-feature access")
    if payload.get("test_features_accessed", False) is not False:
        raise ValueError("basis artifact reports test-feature access")
    target = torch.device(device)
    return SlowPredictiveBasis(
        basis=payload["basis"].float().to(target),
        eigenvalues=payload["eigenvalues"].float().to(target),
        short_lag=int(payload["short_lag"]),
        long_lag=int(payload["long_lag"]),
        regularization=float(payload["regularization"]),
    )


def _as_paths(values: Iterable[str | Path]) -> list[Path]:
    paths = [Path(value) for value in values]
    if not paths:
        raise ValueError("at least one artifact path is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing artifact files: " + ", ".join(missing))
    return paths


def load_utility_bank(
    feature_paths: Iterable[str | Path],
    label_paths: Iterable[str | Path],
    *,
    one_step_weight: float = 0.5,
    device: torch.device | str = "cpu",
) -> UtilityBank:
    """Load the released FIFO/Reservoir bank and construct Eq. (4) targets."""
    features_files = _as_paths(feature_paths)
    labels_files = _as_paths(label_paths)
    if len(features_files) != len(labels_files):
        raise ValueError("feature and label artifact counts must match")
    if not 0.0 <= one_step_weight <= 1.0:
        raise ValueError("one_step_weight must lie in [0,1]")

    feature_rows: list[torch.Tensor] = []
    one_step_rows: list[torch.Tensor] = []
    rollout_rows: list[torch.Tensor] = []
    for feature_path, label_path in zip(
        features_files, labels_files, strict=True
    ):
        feature_payload = torch.load(
            feature_path, map_location="cpu", weights_only=True
        )
        label_payload = torch.load(
            label_path, map_location="cpu", weights_only=True
        )
        features = feature_payload["features"].float()
        tensors = label_payload["tensors"]
        reference = tensors["reference_evict_index"].long()
        one_step = tensors["utility_scalar"].float()
        baseline = one_step.gather(1, reference.unsqueeze(1))
        one_step_delta = one_step - baseline
        rollout_delta = tensors["proposal_rollout_delta"][:, 0].float()
        if features.shape[:2] != one_step_delta.shape:
            raise ValueError(
                f"misaligned feature and label rows: {feature_path}, {label_path}"
            )
        if rollout_delta.shape != one_step_delta.shape:
            raise ValueError(f"misaligned rollout targets: {label_path}")
        feature_rows.append(features)
        one_step_rows.append(one_step_delta)
        rollout_rows.append(rollout_delta)

    features = torch.cat(feature_rows)
    one_step = torch.cat(one_step_rows)
    rollout = torch.cat(rollout_rows)
    one_scale = one_step.reshape(-1).std(unbiased=True).clamp_min(1e-8)
    rollout_scale = rollout.reshape(-1).std(unbiased=True).clamp_min(1e-8)
    relative_costs = (
        float(one_step_weight) * one_step / one_scale
        + (1.0 - float(one_step_weight)) * rollout / rollout_scale
    )
    target = torch.device(device)
    return UtilityBank(
        features=features.to(target),
        relative_costs=relative_costs.to(target),
    )
