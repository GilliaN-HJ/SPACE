#!/usr/bin/env python3
"""Convert the official Assembly101 TSM LMDB into SPACE latent streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm


CAMERA_ID = "C10095_rgb"
FRAME_PATTERN = re.compile(rb"_(\d+)\.jpg$")
RECORDING_PATTERN = re.compile(r"action_both_(\d+)-([^_]+)_")
PARTICIPANTS = {
    "train": {
        "9011", "9013", "9016", "9022", "9024", "9025", "9033",
        "9034", "9035", "9043", "9044", "9046", "9051", "9052",
        "9053", "9055", "9062", "9063", "9064", "9065", "9066",
        "9072", "9074", "9075", "9081", "9082", "9084", "9085",
        "9086",
    },
    "val": {
        "9023", "9036", "9041", "9045", "9054", "9056", "9061",
        "9071", "9076", "9083",
    },
    "test": {
        "9012", "9014", "9015", "9021", "9026", "9031", "9032",
        "9042", "9073",
    },
}


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_recording(recording_id: str) -> tuple[str, str]:
    match = RECORDING_PATTERN.search(recording_id)
    if match is None:
        raise ValueError(f"cannot parse Assembly101 recording id: {recording_id}")
    return match.group(1), match.group(2)


def split_for_participant(participant_id: str) -> str:
    matches = [name for name, values in PARTICIPANTS.items() if participant_id in values]
    if len(matches) != 1:
        raise ValueError(f"participant {participant_id!r} is outside the frozen split")
    return matches[0]


def discover_recordings(environment: lmdb.Environment) -> list[str]:
    suffix = f"/{CAMERA_ID}/".encode("utf-8")
    recordings: set[str] = set()
    with environment.begin(buffers=True) as transaction:
        cursor = transaction.cursor()
        for key in cursor.iternext(keys=True, values=False):
            raw = bytes(key)
            position = raw.find(suffix)
            if position > 0:
                recordings.add(raw[:position].decode("utf-8"))
    if not recordings:
        raise ValueError(
            f"no {CAMERA_ID} entries were found; check that --lmdb points to "
            "the extracted official TSM feature database"
        )
    return sorted(recordings)


def read_recording(
    environment: lmdb.Environment,
    recording_id: str,
    frame_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"{recording_id}/{CAMERA_ID}/{CAMERA_ID}_".encode("utf-8")
    frames: list[int] = []
    features: list[np.ndarray] = []
    with environment.begin(buffers=True) as transaction:
        cursor = transaction.cursor()
        present = cursor.set_range(prefix)
        while present:
            key = bytes(cursor.key())
            if not key.startswith(prefix):
                break
            match = FRAME_PATTERN.search(key)
            if match is None:
                raise ValueError(f"malformed Assembly101 TSM key: {key!r}")
            value = np.frombuffer(cursor.value(), dtype=np.float32)
            if value.shape != (2048,):
                raise ValueError(f"expected a 2048-D TSM feature, got {value.shape}")
            frames.append(int(match.group(1)))
            features.append(value.copy())
            present = cursor.next()
    if not features:
        raise KeyError(f"no TSM features found for {recording_id}/{CAMERA_ID}")
    order = np.argsort(np.asarray(frames))
    frame_array = np.asarray(frames, dtype=np.int64)[order][::frame_stride]
    feature_array = np.stack(features)[order][::frame_stride]
    if len(np.unique(frame_array)) != len(frame_array):
        raise ValueError(f"duplicate frame ids in {recording_id}")
    feature_array /= np.maximum(
        np.linalg.norm(feature_array, axis=1, keepdims=True), 1e-8
    )
    return torch.from_numpy(feature_array), torch.from_numpy(frame_array)


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val"),
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--minimum-valid-latents", type=int, default=120)
    args = parser.parse_args()

    requested = set(args.splits)
    if "test" in requested and not args.allow_test:
        raise ValueError("test conversion requires the explicit --allow-test flag")
    if args.frame_stride <= 0 or args.frame_rate <= 0:
        raise ValueError("frame stride and frame rate must be positive")

    environment = lmdb.open(
        str(args.lmdb.resolve()),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=32,
        subdir=args.lmdb.is_dir(),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    dropped: list[dict] = []
    try:
        recordings = discover_recordings(environment)
        selected = []
        for recording_id in recordings:
            participant_id, toy_id = parse_recording(recording_id)
            split = split_for_participant(participant_id)
            if split in requested:
                selected.append((recording_id, participant_id, toy_id, split))
        for recording_id, participant_id, toy_id, split in tqdm(
            selected, desc="Assembly101 TSM"
        ):
            features, frames = read_recording(
                environment, recording_id, args.frame_stride
            )
            if len(features) < args.minimum_valid_latents:
                dropped.append(
                    {
                        "video_id": recording_id,
                        "split": split,
                        "valid_length": len(features),
                    }
                )
                continue
            preprocess = {
                "camera_id": CAMERA_ID,
                "frame_stride": args.frame_stride,
                "frame_rate": args.frame_rate,
                "l2_normalize": True,
                "participant_split": "assembly101_participant_disjoint_v1",
            }
            destination = output / f"{recording_id}.pt"
            atomic_torch_save(
                {
                    "video_id": recording_id,
                    "latents": features.float(),
                    "clip_indices": frames.long(),
                    "timestamps_sec": frames.float() / args.frame_rate,
                    "valid_length": len(features),
                    "encoder_id": "Assembly101 official TSM 2048-D features",
                    "encoder_checkpoint_hash": "official_assembly101_tsm_release",
                    "preprocess_hash": canonical_hash(preprocess),
                    "metadata": {
                        "participant_id": participant_id,
                        "toy_id": toy_id,
                        "split": split,
                        "view_id": "v1",
                        "camera_id": CAMERA_ID,
                    },
                },
                destination,
            )
            rows.append(
                {
                    "video_id": recording_id,
                    "path": destination.name,
                    "split": split,
                    "valid_length": len(features),
                    "subject_id": f"Assembly101-P{participant_id}",
                    "toy_id": toy_id,
                    "view_id": "v1",
                }
            )
    finally:
        environment.close()

    manifest = {
        "format_version": 1,
        "dataset": "Assembly101",
        "protocol_id": "assembly101_single_view_participant_disjoint_v1",
        "d_in": 2048,
        "preprocess": {
            "source": "Assembly101 official TSM features",
            "camera_id": CAMERA_ID,
            "frame_stride": args.frame_stride,
            "frame_rate": args.frame_rate,
            "effective_stride_sec": args.frame_stride / args.frame_rate,
            "l2_normalize": True,
            "minimum_valid_latents": args.minimum_valid_latents,
        },
        "requested_splits": sorted(requested),
        "test_features_accessed": "test" in requested,
        "videos": rows,
        "dropped": dropped,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "records": len(rows),
                "dropped": len(dropped),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
