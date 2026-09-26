#!/usr/bin/env python3
"""Fit SPACE's train-only slow predictive basis without action labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from memory_jepa.data.latent_dataset import LatentCorpus
from memory_jepa.memory.metastable import fit_metastable_subspace
from memory_jepa.memory.policies.counter_based import (
    CounterBasedReservoirSamplingPolicy,
    stable_video_seed,
)
from memory_jepa.memory.state import MemoryState
from memory_jepa.utils.io import atomic_torch_save, write_json
from memory_jepa.utils.modeling import load_predictor_checkpoint
from training_utils import (
    candidate_tensors,
    insert_first_free,
    online_capacity,
    predict_state,
    replace_compact,
)


@torch.no_grad()
def extract_record(
    row: dict,
    corpus: LatentCorpus,
    predictor: Any,
    *,
    device: torch.device,
    precision: str,
    policy_seed: int,
) -> torch.Tensor:
    record = corpus.load(row)
    length = int(record["valid_length"])
    latents = record["latents"][:length].float().to(device)
    context = int(predictor.local_context_len)
    maximum_horizon = max(int(value) for value in predictor.horizons)
    capacity = online_capacity(predictor)
    state = MemoryState.empty(capacity, latents.shape[-1], device)
    sampler = CounterBasedReservoirSamplingPolicy(
        capacity,
        stable_video_seed(str(row["video_id"]), int(policy_seed)),
    )
    projected = []
    for event_time in range(length - maximum_horizon - context):
        current_time = event_time + context
        event = latents[event_time]
        local = latents[current_time - context + 1 : current_time + 1]
        if state.active_count < capacity:
            state = insert_first_free(state, event, event_time)
        else:
            values, timestamps, event_ids = candidate_tensors(
                state, event, event_time
            )
            action = int(sampler.select_evict_index(
                values.unsqueeze(0),
                timestamps.unsqueeze(0),
                torch.ones(
                    1, capacity + 1, dtype=torch.bool, device=device
                ),
                local.unsqueeze(0),
                torch.tensor([current_time], device=device),
                candidate_event_ids=event_ids.unsqueeze(0),
            ).item())
            state = replace_compact(state, event, event_time, action)
        if state.active_count == capacity:
            prediction = predict_state(
                predictor, state, local, current_time, precision
            )
            projected.append(
                predictor.project_inputs(prediction, detach=True).to(torch.float16)
            )
    if not projected:
        raise ValueError(f"recording {row['video_id']} has no full-memory state")
    return torch.stack(projected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--predictor-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = protocol["slow_basis"]
    policy_seed = (
        int(args.predictor_seed)
        if args.policy_seed is None
        else int(args.policy_seed)
    )
    device = torch.device(args.device)
    corpus = LatentCorpus(args.manifest)
    if corpus.manifest.get("dataset") != "Assembly101":
        raise ValueError("slow-basis fitting requires Assembly101")
    if corpus.manifest.get("test_features_accessed") is not False:
        raise ValueError("slow-basis fitting refuses test features")
    train_records = corpus.split_records("train")
    participants = sorted({str(row["subject_id"]) for row in train_records})
    if len(participants) != int(config["training_participants"]):
        raise ValueError("unexpected number of training participants")

    predictor, checkpoint = load_predictor_checkpoint(args.predictor, device)
    if int(checkpoint["seed"]) != int(args.predictor_seed):
        raise ValueError("predictor checkpoint seed differs from --predictor-seed")
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    state_count = 0
    for index, row in enumerate(train_records):
        trajectory = extract_record(
            row,
            corpus,
            predictor,
            device=device,
            precision=args.precision,
            policy_seed=policy_seed,
        )
        state_count += len(trajectory)
        for horizon_index in range(trajectory.shape[1]):
            grouped[str(row["subject_id"])].append(trajectory[:, horizon_index])
        print(json.dumps({
            "record": index + 1,
            "records": len(train_records),
            "video_id": row["video_id"],
            "states": len(trajectory),
        }), flush=True)

    subspace = fit_metastable_subspace(
        [grouped[participant] for participant in participants],
        short_lag=int(config["short_lag"]),
        long_lag=int(config["long_lag"]),
        rank=int(config["rank"]),
        regularization=float(config["regularization"]),
    )
    artifact = {
        "basis": subspace.basis.cpu(),
        "eigenvalues": subspace.eigenvalues.cpu(),
        "short_lag": subspace.short_lag,
        "long_lag": subspace.long_lag,
        "regularization": subspace.regularization,
        "rank": subspace.rank,
        "input_dimension": subspace.input_dimension,
        "horizons": [int(value) for value in predictor.horizons],
        "predictor_hash": str(checkpoint["predictor_hash"]),
        "protocol_id": str(protocol["protocol_id"]),
        "fit_split": "train",
        "reference_policy": "counter_based_reservoir",
        "reference_policy_seed": policy_seed,
        "participants": participants,
        "validation_features_accessed": False,
        "test_features_accessed": False,
    }
    atomic_torch_save(artifact, args.output)
    write_json({
        "protocol_id": protocol["protocol_id"],
        "output": str(args.output.resolve()),
        "participants": len(participants),
        "recordings": len(train_records),
        "predictive_states": state_count,
        "rank": subspace.rank,
        "short_lag": subspace.short_lag,
        "long_lag": subspace.long_lag,
        "eigenvalues": subspace.eigenvalues.cpu().tolist(),
        "elapsed_sec": time.time() - started,
        "fit_split": "train",
        "validation_features_accessed": False,
        "test_features_accessed": False,
    }, args.summary_output)


if __name__ == "__main__":
    main()
