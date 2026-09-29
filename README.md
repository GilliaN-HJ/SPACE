# SPACE: Sparse Predictive Attractor via Counterfactual Eviction

**Paper:** [arXiv:2609.32592](https://arxiv.org/abs/2609.32592)

## Installation

The reference environment uses Python 3.10, PyTorch 2.5.1, and CUDA 12.1.

```bash
conda create -n space python=3.10 -y
conda activate space
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Download Assembly101 features

Download and extract the official TSM features, no action annotations are needed:

```bash
hf download cvml-nus/assembly101 TSM_features/C10095_rgb.zip \
  --repo-type dataset --local-dir data/assembly101_raw
unzip data/assembly101_raw/TSM_features/C10095_rgb.zip \
  -d data/assembly101_raw/C10095_rgb
```

## Train SPACE

Train the default seed with:

```bash
python scripts/train_assembly101.py \
  --lmdb <path-to-C10095_rgb-LMDB> \
  --gpus 0
```

To train more predictor/policy seeds:

```bash
python scripts/train_assembly101.py \
  --lmdb <path-to-C10095_rgb-LMDB> \
  --seeds <seed1> <seed2> <seed3> \
  --gpus 0 1 2
```

The pipeline is resumable and skips completed artifacts. Stages can also be run separately:

```bash
python scripts/train_assembly101.py \
  --seeds 0 \
  --gpus 0 \
  --stages predictors slow_basis banks
```

Training writes:

```text
outputs/assembly101_predictors/seed_<seed>/predictor_best.pt
outputs/space_training/seed_<seed>/slow_basis.pt
outputs/space_training/seed_<seed>/utility_banks/
outputs/space_training/artifacts.json
```


The trained artifacts can be loaded directly by the public controller:

```python
from space import SPACE, load_config, load_predictor_checkpoint
from space import load_slow_basis, load_utility_bank

config = load_config("configs/space_k16.yaml")
predictor, _ = load_predictor_checkpoint(
    "outputs/assembly101_predictors/seed_0/predictor_best.pt", "cuda"
)
basis = load_slow_basis(
    "outputs/space_training/seed_0/slow_basis.pt", "cuda"
)
bank_root = "outputs/space_training/seed_0/utility_banks"
bank = load_utility_bank(
    [f"{bank_root}/fifo_features.pt", f"{bank_root}/reservoir_features.pt"],
    [f"{bank_root}/fifo_r32_delta.pt", f"{bank_root}/reservoir_r32_delta.pt"],
    one_step_weight=0.5,
    device="cuda",
)
controller = SPACE(
    config=config,
    predictor=predictor,
    slow_predictive_basis=basis,
    utility_bank=bank,
    policy_seed=0,
    device="cuda",
)
```

## Repository layout

```text
configs/          Reported model and controller settings
src/space/        Compact dataset-independent SPACE implementation
src/memory_jepa/  Predictor and training components
scripts/          Assembly101 data preparation and artifact training
```
