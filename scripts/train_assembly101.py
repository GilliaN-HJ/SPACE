#!/usr/bin/env python3
"""Train all SPACE artifacts from official Assembly101 TSM features."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


STAGES = ("data", "predictors", "slow_basis", "banks")


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_parallel(commands: list[list[str]], *, cwd: Path, workers: int) -> None:
    if not commands:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run, command, cwd=cwd) for command in commands]
        for future in futures:
            future.result()


def require_files(paths: list[Path], message: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(message + ": " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be nonempty and unique")
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus must be nonempty and unique")

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    python = sys.executable
    selected = set(args.stages)
    manifest = root / "data/latents/assembly101_tsm_v1_stride15/manifest.json"
    protocol = root / "configs/assembly101_from_scratch.json"
    training_root = root / "outputs/space_training"

    if "data" in selected and not manifest.is_file():
        if args.lmdb is None:
            raise ValueError("--lmdb is required for the data stage")
        run([
            python,
            str(scripts / "prepare_assembly101_tsm.py"),
            "--lmdb", str(args.lmdb.resolve()),
            "--output-dir", str(manifest.parent),
        ], cwd=root)
    require_files([manifest], "prepared Assembly101 manifest is missing")

    predictor_paths = {
        seed: root / f"outputs/assembly101_predictors/seed_{seed}/predictor_best.pt"
        for seed in args.seeds
    }
    if "predictors" in selected:
        commands = []
        for index, seed in enumerate(args.seeds):
            if predictor_paths[seed].is_file():
                continue
            commands.append([
                python,
                str(scripts / "train_predictor.py"),
                "--config", str(root / "configs/assembly101.yaml"),
                "--manifest", str(manifest),
                "--seed", str(seed),
                "--device", f"cuda:{args.gpus[index % len(args.gpus)]}",
                "--output", str(root / "outputs/assembly101_predictors"),
            ])
        run_parallel(
            commands, cwd=root, workers=min(len(args.gpus), len(commands))
        )
    require_files(list(predictor_paths.values()), "predictor training is incomplete")

    slow_paths = {
        seed: training_root / f"seed_{seed}/slow_basis.pt" for seed in args.seeds
    }
    if "slow_basis" in selected:
        commands = []
        for index, seed in enumerate(args.seeds):
            if slow_paths[seed].is_file():
                continue
            commands.append([
                python,
                str(scripts / "fit_assembly101_slow_basis.py"),
                "--protocol", str(protocol),
                "--manifest", str(manifest),
                "--predictor", str(predictor_paths[seed]),
                "--predictor-seed", str(seed),
                "--policy-seed", str(seed),
                "--device", f"cuda:{args.gpus[index % len(args.gpus)]}",
                "--precision", "bf16",
                "--output", str(slow_paths[seed]),
                "--summary-output", str(
                    slow_paths[seed].with_name("slow_basis.json")
                ),
            ])
        run_parallel(
            commands, cwd=root, workers=min(len(args.gpus), len(commands))
        )
    require_files(list(slow_paths.values()), "slow-basis fitting is incomplete")

    bank_dirs = {
        seed: training_root / f"seed_{seed}/utility_banks" for seed in args.seeds
    }
    if "banks" in selected:
        commands = []
        for index, seed in enumerate(args.seeds):
            required = [
                bank_dirs[seed] / "fifo_features.pt",
                bank_dirs[seed] / "reservoir_features.pt",
                bank_dirs[seed] / "fifo_r32_delta.pt",
                bank_dirs[seed] / "reservoir_r32_delta.pt",
            ]
            if all(path.is_file() for path in required):
                continue
            commands.append([
                python,
                str(scripts / "build_assembly101_utility_banks.py"),
                "--protocol", str(protocol),
                "--predictor", str(predictor_paths[seed]),
                "--manifest", str(manifest),
                "--output-dir", str(bank_dirs[seed]),
                "--device", f"cuda:{args.gpus[index % len(args.gpus)]}",
            ])
        run_parallel(
            commands, cwd=root, workers=min(len(args.gpus), len(commands))
        )

    summary = {
        "manifest": str(manifest),
        "seeds": {
            str(seed): {
                "predictor": str(predictor_paths[seed]),
                "slow_basis": str(slow_paths[seed]),
                "utility_banks": str(bank_dirs[seed]),
            }
            for seed in args.seeds
        },
    }
    training_root.mkdir(parents=True, exist_ok=True)
    (training_root / "artifacts.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
