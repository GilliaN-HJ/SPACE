#!/usr/bin/env python3
"""Build the train-only FIFO and Reservoir utility banks from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from training_utils import (
    candidate_tensors,
    insert_first_free,
    online_features,
    proposal_rollout_losses,
    replace_compact,
    reservoir_action,
)
from memory_jepa.data.latent_dataset import LatentCorpus
from memory_jepa.memory.policies.baselines import FIFOPolicy
from memory_jepa.memory.policies.counter_based import stable_video_seed
from memory_jepa.memory.state import MemoryState
from memory_jepa.oracle.counterfactual import exact_counterfactual_utility
from memory_jepa.oracle.label_store import UtilityLabelWriter
from memory_jepa.utils.hashing import file_hash
from memory_jepa.utils.io import atomic_torch_save, write_json
from memory_jepa.utils.modeling import load_predictor_checkpoint


EXTRA_KEYS = (
    "proposal_evict_indices",
    "proposal_rollout_loss_r8",
    "proposal_rollout_loss_r32",
    "proposal_rollout_per_horizon_r8",
    "proposal_rollout_per_horizon_r32",
    "video_index",
    "subject_index",
    "time_index",
    "proposal_rollout_delta",
    "reference_evict_index",
)


def subject_seed(seed: int, subject: str) -> int:
    value = hashlib.blake2b(
        f"{int(seed)}\0{subject}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(value, "little") & ((1 << 63) - 1)


def select_times(
    records: list[dict],
    *,
    context: int,
    capacity: int,
    maximum_horizon: int,
    rollout_steps: int,
    states_per_participant: int,
    selection_seed: int,
) -> tuple[dict[str, set[int]], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[str(row["subject_id"])].append(row)
    selected: dict[str, set[int]] = {}
    audit = {}
    for subject in sorted(grouped):
        rows = sorted(grouped[subject], key=lambda row: str(row["video_id"]))
        eligible: list[np.ndarray] = []
        for row in rows:
            first = context + capacity
            last = int(row["valid_length"]) - maximum_horizon - rollout_steps
            if last < first:
                eligible.append(np.empty(0, dtype=np.int64))
            else:
                eligible.append(np.arange(first, last + 1, dtype=np.int64))
        valid_indices = [index for index, times in enumerate(eligible) if len(times)]
        if not valid_indices:
            raise ValueError(f"participant {subject} has no rollout-valid states")
        rng = np.random.default_rng(subject_seed(selection_seed, subject))
        assigned = rng.choice(valid_indices, size=states_per_participant, replace=True)
        counts = Counter(int(index) for index in assigned.tolist())
        subject_rows = []
        for index in valid_indices:
            count = counts[index]
            if count > len(eligible[index]):
                raise ValueError(
                    f"recording {rows[index]['video_id']} has too few unique states"
                )
            times = rng.choice(eligible[index], size=count, replace=False)
            values = {int(value) for value in times.tolist()}
            selected[str(rows[index]["video_id"])] = values
            subject_rows.append(
                {
                    "video_id": str(rows[index]["video_id"]),
                    "eligible": int(len(eligible[index])),
                    "selected": sorted(values),
                }
            )
        audit[subject] = subject_rows
    return selected, audit


def update_state(
    state: MemoryState,
    policy: str,
    event: torch.Tensor,
    event_time: int,
    local: torch.Tensor,
    current_time: int,
    video_id: str,
    behavior_seed: int,
) -> MemoryState:
    if state.active_count < state.capacity:
        return insert_first_free(state, event, event_time)
    values, timestamps, event_ids = candidate_tensors(state, event, event_time)
    if policy == "fifo":
        action = int(
            FIFOPolicy().select_evict_index(
                values.unsqueeze(0),
                timestamps.unsqueeze(0),
                torch.ones(
                    1, len(values), dtype=torch.bool, device=values.device
                ),
                local.unsqueeze(0),
                torch.tensor([current_time], device=values.device),
                candidate_event_ids=event_ids.unsqueeze(0),
            ).item()
        )
    else:
        action = reservoir_action(
            values,
            timestamps,
            event_ids,
            local,
            current_time,
            capacity=state.capacity,
            video_id=video_id,
            policy_seed=behavior_seed,
        )
    return replace_compact(state, event, event_time, action)


@torch.no_grad()
def build_policy_bank(
    policy: str,
    *,
    selected: dict[str, set[int]],
    records: list[dict],
    corpus: LatentCorpus,
    predictor,
    checkpoint: dict,
    config: dict,
    manifest_path: Path,
    device: torch.device,
    output_dir: Path,
) -> dict:
    context = int(predictor.local_context_len)
    capacity = int(config["capacity"])
    horizons = tuple(int(value) for value in predictor.horizons)
    weights = torch.tensor(
        checkpoint["config"]["predictor"]["horizon_weights"], device=device
    )
    loss_type = str(checkpoint["config"]["predictor"]["target_loss"])
    precision = str(config["precision"])
    behavior_seed = int(config["reservoir_behavior_seed"])
    reference_seed = int(config["reservoir_reference_seed"])
    video_lookup = [str(row["video_id"]) for row in records]
    video_index = {value: index for index, value in enumerate(video_lookup)}
    subject_lookup = sorted({str(row["subject_id"]) for row in records})
    subject_index = {value: index for index, value in enumerate(subject_lookup)}
    writer = UtilityLabelWriter(extra_keys=EXTRA_KEYS)
    features: list[torch.Tensor] = []

    for row in tqdm(records, desc=f"utility bank: {policy}"):
        video_id = str(row["video_id"])
        chosen = selected.get(video_id, set())
        if not chosen:
            continue
        record = corpus.load(row)
        length = int(record["valid_length"])
        latents = record["latents"][:length].float().to(device)
        state = MemoryState.empty(capacity, latents.shape[-1], device)
        for current_time in range(context, max(chosen) + 1):
            event_time = current_time - context
            event = latents[event_time]
            local = latents[current_time - context + 1 : current_time + 1]
            if state.active_count == capacity and current_time in chosen:
                candidate, timestamps, event_ids = candidate_tensors(
                    state, event, event_time
                )
                age = (current_time - timestamps).float()
                valid = torch.ones(
                    1, capacity + 1, dtype=torch.bool, device=device
                )
                local_age = torch.arange(
                    context - 1, -1, -1, device=device, dtype=torch.float32
                )
                targets = torch.stack(
                    [latents[current_time + horizon] for horizon in horizons]
                )
                exact = exact_counterfactual_utility(
                    predictor,
                    local.unsqueeze(0),
                    candidate.unsqueeze(0),
                    valid,
                    targets.unsqueeze(0),
                    torch.ones(
                        1, len(horizons), dtype=torch.bool, device=device
                    ),
                    weights,
                    loss_type,
                    local_age.unsqueeze(0),
                    age.unsqueeze(0),
                )
                proposals = torch.arange(capacity + 1, device=device).unsqueeze(0)
                scalar, per_horizon = proposal_rollout_losses(
                    predictor,
                    latents,
                    current_time,
                    candidate,
                    timestamps,
                    proposals,
                    horizons,
                    weights,
                    loss_type,
                    context,
                    stable_video_seed(video_id, reference_seed),
                )
                reference = reservoir_action(
                    candidate,
                    timestamps,
                    event_ids,
                    local,
                    current_time,
                    capacity=capacity,
                    video_id=video_id,
                    policy_seed=reference_seed,
                )
                delta = scalar[32] - scalar[32][:, reference : reference + 1]
                features.append(
                    online_features(
                        predictor,
                        candidate,
                        age,
                        local,
                        device,
                        precision,
                    ).cpu()
                )
                writer.append(
                    local=local,
                    candidate_memory=candidate,
                    candidate_age=age,
                    candidate_valid=valid[0],
                    utility_per_horizon=exact.utility_per_horizon[0],
                    utility_scalar=exact.utility_scalar[0],
                    oracle_evict_index=exact.oracle_evict_index[0],
                    proposal_evict_indices=proposals,
                    proposal_rollout_loss_r8=scalar[8],
                    proposal_rollout_loss_r32=scalar[32],
                    proposal_rollout_per_horizon_r8=per_horizon[8],
                    proposal_rollout_per_horizon_r32=per_horizon[32],
                    video_index=torch.tensor(video_index[video_id]),
                    subject_index=torch.tensor(subject_index[str(row["subject_id"])]),
                    time_index=torch.tensor(current_time),
                    proposal_rollout_delta=delta,
                    reference_evict_index=torch.tensor(reference),
                )
            state = update_state(
                state,
                policy,
                event,
                event_time,
                local,
                current_time,
                video_id,
                behavior_seed,
            )

    state_policy = (
        "fifo" if policy == "fifo" else f"counter_based_reservoir_seed_{behavior_seed}"
    )
    metadata = {
        "format_version": 1,
        "predictor_hash": str(checkpoint["predictor_hash"]),
        "config_hash": str(checkpoint["config_hash"]),
        "manifest": str(manifest_path.resolve()),
        "manifest_hash": file_hash(manifest_path),
        "split": "train",
        "state_policy": state_policy,
        "sampling": "participant_balanced_uniform_record_then_time",
        "states_per_participant": int(config["states_per_participant"]),
        "selection_seed": int(config["selection_seed"]),
        "memory_budget": capacity,
        "local_context_len": context,
        "states": len(writer),
        "video_lookup": video_lookup,
        "subject_lookup": subject_lookup,
        "reference_policy": "counter_based_reservoir",
        "reference_policy_seed": reference_seed,
        "common_random_numbers_across_branches": True,
        "rollout_lengths": [8, 32],
        "validation_features_accessed": False,
        "test_features_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / f"{policy}_r32_delta.pt"
    writer.save(label_path, metadata)
    feature_metadata = dict(
        metadata,
        source_labels=str(label_path.resolve()),
        source_labels_sha256=file_hash(label_path),
    )
    feature_tensor = torch.stack(features)
    target_tensor = torch.stack(writer.rows["proposal_rollout_delta"])[:, 0]
    atomic_torch_save(
        {
            "features": feature_tensor,
            "targets": target_tensor,
            "metadata": feature_metadata,
        },
        output_dir / f"{policy}_features.pt",
    )
    return {
        "policy": policy,
        "states": len(writer),
        "feature_shape": list(feature_tensor.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = protocol["utility_bank"]
    device = torch.device(args.device)
    predictor, checkpoint = load_predictor_checkpoint(str(args.predictor), device)
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    if int(config["capacity"]) != predictor.memory_slots - 1:
        raise ValueError("bank capacity differs from the predictor checkpoint")
    corpus = LatentCorpus(args.manifest)
    if corpus.manifest.get("test_features_accessed") is not False:
        raise ValueError("utility-bank construction refuses test features")
    records = corpus.split_records("train")
    selected, audit = select_times(
        records,
        context=int(predictor.local_context_len),
        capacity=int(config["capacity"]),
        maximum_horizon=max(int(value) for value in predictor.horizons),
        rollout_steps=int(config["rollout_steps"]),
        states_per_participant=int(config["states_per_participant"]),
        selection_seed=int(config["selection_seed"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "protocol_id": protocol["protocol_id"],
            "selection_seed": int(config["selection_seed"]),
            "states_per_participant": int(config["states_per_participant"]),
            "participants": audit,
        },
        args.output_dir / "sample_times.json",
    )
    result = [
        build_policy_bank(
            policy,
            selected=selected,
            records=records,
            corpus=corpus,
            predictor=predictor,
            checkpoint=checkpoint,
            config=config,
            manifest_path=args.manifest,
            device=device,
            output_dir=args.output_dir,
        )
        for policy in ("fifo", "reservoir")
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
