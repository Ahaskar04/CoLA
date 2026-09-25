# CoLA: Coordination via Learned Adapters

CoLA coordinates several robot arms on top of a frozen vision-language-action
backbone. Each arm reads the backbone's features for its own camera, encodes
them into a short message (64-d by default) for its partners, decodes the
messages it receives, and predicts a chunk of joint targets with a diffusion
U-Net head. Only the message encoders/decoders and the head are trained; the
backbone (Octo-Base, or pi0.5 in the backbone study) stays frozen, so its
features are extracted once and cached.

## Tasks

All tasks are MuJoCo scenes with ALOHA ViperX arms. Demonstrations come from
scripted experts driven by inverse kinematics (mink). In all experiments,
episodes start with arm A already holding the box.

| Task | Setup | Success |
|---|---|---|
| `handover_2arm` | A hands the box to B. | B holds the lifted box on its own. |
| `handover_3arm` | A hands the box to B; B turns and hands it to C. | C holds the box on its own. |
| `handover_marker` | A coloured marker on the box is visible to A only. After the handover, B must drop the box into the tray of that colour. | Correct tray, among completed handovers (chance 1 in 3). |

## Repository layout

```
cola/              CoLA model and datasets (two- and three-arm)
data_collection/   scripted expert and demo recorder, one folder per task
envs/              MuJoCo scenes, one folder per task; shared meshes in envs/assets
preprocess/        demo -> training cache, and frozen Octo-Base feature extraction (main.py runs both)
train/             CoLA training (two- and three-arm)
eval/              closed-loop evaluation in simulation, one script per task
analysis/          message probes and the message bank for the swap test
baselines/octo/    Octo-Base finetuning baselines and zero-shot Octo
pi05/              pi0.5 backbone study (separate environment, see below)
```

Outputs go to `data/`, `runs/` and `results/` inside the repository (all
git-ignored). Each step's default input is the previous step's default output:

```
data/demos/<task>  ->  data/cache/<task>  ->  data/features/<task>  ->  runs/<run>  ->  results/
```

## Installation

Main environment (CoLA, Octo baselines, data collection), Python 3.10 or 3.11
(TensorFlow 2.15 has no 3.12 build) with a CUDA GPU:

```bash
git clone https://github.com/octo-models/octo.git
pip install -e octo
pip install -r octo/requirements.txt   # Octo's own dependencies
pip install -r requirements.txt
pip install -r requirements-cuda.txt   # JAX 0.4.20 on CUDA 12.3 / cuDNN 8.9
pip install numpy==1.24.3              # last: jaxlib 0.4.20 needs the numpy 1.x ABI
```

The Octo-base weights download from Hugging Face on first use; on GPU nodes
without internet access, load the model once on a machine that has it and set
`HF_HUB_OFFLINE=1`. For headless rendering set `export MUJOCO_GL=egl`. All
commands below run from the repository root.

## Reproducing the experiments

### 1. Collect demonstrations

`COLA_HANDOVER_ONLY=1` starts every episode with arm A holding the box, which is
the setting used throughout. Without it the expert also picks the box up and
the output goes to `data/demos/<task>_fulltask/`.

```bash
export COLA_HANDOVER_ONLY=1
COLA_NUM_EPISODES=300 python data_collection/handover_2arm/collect_demos.py
COLA_NUM_EPISODES=300 python data_collection/handover_3arm/collect_demos.py
COLA_NUM_EPISODES=150 python data_collection/handover_marker/collect_demos.py
```

Each episode is saved as `episode_XXXX.h5` together with a review video.

### 2. Build caches and extract frozen Octo-Base features

```bash
python preprocess/main.py handover_2arm      # 150 episodes, sampled with seed 42; wrist + overhead cameras
python preprocess/main.py handover_marker    # wrist cameras only
python preprocess/main.py handover_3arm      # 130 episodes, sampled with seed 42
```

Each command reads `data/demos/<task>`, writes the training cache to
`data/cache/<task>` and the Octo-Base features to `data/features/<task>`. Splits
are 80/10/10 by episode. The overhead camera, which the overhead-only runs below
use, is extracted for the two-arm and three-arm tasks. Flags override the
defaults: `--data_dir`, `--cache_dir` and `--feat_dir` for the folders,
`--overhead`/`--no-overhead`, `--max_episodes` and `--seed`. `--steps cache` or
`--steps features` runs one stage; feature extraction needs a GPU.

### 3. Train CoLA

Every run uses the same recipe, `$ARGS`:

```bash
ARGS="--lambda_gripper 0.25 --grip_transition_weight 0 --diffusion --diffusion_unet --num_epochs 150"

# Two-arm: overhead camera, or wrist cameras
python train/train_2arm.py $ARGS --overhead_only --run_dir runs/handover_2arm
python train/train_2arm.py $ARGS --run_dir runs/handover_2arm_wrist

# Hidden marker: wrist cameras
python train/train_2arm.py $ARGS --cache_dir data/cache/handover_marker \
    --feat_dir data/features/handover_marker --run_dir runs/handover_marker

# Three-arm: overhead camera
python train/train_3arm.py $ARGS --overhead_only --run_dir runs/handover_3arm
```

Each run has a no-message control: add `--no_messages` and name it
`<run>_nomsg`. It is identical except that the channel is severed. `--seed` sets
the training seed (default 0). The best checkpoint (lowest validation loss) is
saved to `<run_dir>/checkpoints/best_model.pkl`.

### 4. Evaluate in simulation

Episodes are seeded by their index, so every model sees the same start states.
Where we report several blocks of episodes, they are `--seed-offset` 0, 1000,
2000, 3000 and 4000; the commands below run block 0. Evaluate a `_nomsg` run
with `--no-messages` (the evaluator refuses a mismatch), and add `--no-videos`
to skip the rollout videos.

```bash
# Two-arm: 200 episodes, starting from the handover
python eval/eval_2arm.py --model runs/handover_2arm/checkpoints/best_model.pkl \
    --handover-only --episodes 200 --seed-offset 0 --results-path results/handover_2arm/seed0.json

# Hidden marker: 150 episodes, 50 per colour
python eval/eval_marker.py --model runs/handover_marker/checkpoints/best_model.pkl \
    --episodes 150 --seed-offset 0 --results-path results/handover_marker/seed0.json

# Three-arm: 200 episodes; success means C ends up holding the box
python eval/eval_3arm.py --model runs/handover_3arm/checkpoints/best_model.pkl \
    --episodes 200 --seed-offset 0 --results-path results/handover_3arm/seed0.json
```

Each evaluator prints a summary and writes it, with per-episode records, to
`--results-path`. In the two-arm task, an episode whose scripted start loses the
grasp is skipped: it is left out of the rates and counted under `skipped`.

### 5. Octo baselines

Octo-Base is finetuned with a U-Net head, a batch of 128 and the same number of
gradient steps as the matching CoLA run. The decentralised baseline trains one
policy per arm (`--arm a`, `b`, and `c` for three-arm) with no channel between
them; `--arm both` trains a single centralised policy. `--tune head_only` keeps
the backbone frozen, as in CoLA; without it the whole model is finetuned.
`--resume` continues an interrupted run. The commands below train arm A; change
`--arm` and `--save-dir` for the others.

```bash
OCTO="--pretrained-path hf://rail-berkeley/octo-base --head unet --action-horizon 12 --batch-size 128 --grad-accum 2"

# Two-arm, overhead camera (wrist rows: --image-key image_wrist_a, or image_wrist_b for arm B)
python baselines/octo/finetune_octo.py $OCTO --manifest data/cache/handover_2arm/split_manifest.json \
    --steps 12600 --image-key image_overhead --arm a --save-dir runs/octo/ho_a

# Hidden marker: each arm uses its own wrist camera
python baselines/octo/finetune_octo.py $OCTO --manifest data/cache/handover_marker/split_manifest.json \
    --steps 20250 --arm a --save-dir runs/octo/marker_a

# Three-arm, overhead camera
python baselines/octo/finetune_octo.py $OCTO --manifest data/cache/handover_3arm/split_manifest.json \
    --steps 26850 --image-key image_overhead --arm a --save-dir runs/octo/3arm_a
```

The Octo evaluators use CoLA's scoring code, and check at startup that it has
not drifted. For a centralised policy, pass `--centralised` and only
`--checkpoint-a`; the two-arm wrist rows add `--scene-xml envs/handover_2arm/scene.xml`.

```bash
python baselines/octo/eval_octo_2arm.py --checkpoint-a runs/octo/ho_a --checkpoint-b runs/octo/ho_b \
    --handover-only --episodes 200 --seed-offset 0 --results-path results/octo/ho_seed0.json
python baselines/octo/eval_octo_marker.py --checkpoint-a runs/octo/marker_a --checkpoint-b runs/octo/marker_b \
    --episodes 150 --results-path results/octo/marker.json
python baselines/octo/eval_octo_3arm.py --checkpoint-a runs/octo/3arm_a --checkpoint-b runs/octo/3arm_b \
    --checkpoint-c runs/octo/3arm_c --episodes 200 --results-path results/octo/3arm.json

# Zero-shot Octo-Base on the two-arm scene, including the pick-up
python baselines/octo/eval_octo_zeroshot.py --octo-checkpoint octo-base --episodes 200
```

### 6. pi0.5 backbone study (hidden marker)

These scripts use the hidden-marker cache from step 2 and run in a separate
environment with [openpi](https://github.com/Physical-Intelligence/openpi),
installed as described in its README. openpi pins an older MuJoCo, so install
MuJoCo 3.12.0 (the version used for every other evaluation) into its own
directory and put it first on `PYTHONPATH` for the evaluation scripts only. The
pi0.5 base checkpoint is read from openpi's cache (`~/.cache/openpi` unless
`OPENPI_DATA_HOME` is set):

```bash
uv pip install h5py imageio imageio-ffmpeg tqdm
uv pip install --no-deps --target ~/mujoco312 mujoco==3.12.0
python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')"
```

**CoLA on frozen pi0.5.** CoLA's adapters, channel and U-Net head on pooled
PaliGemma prefix features (2048-d) instead of Octo-Base's. For the control, add
`--no_messages` to training; the evaluator reads the setting from the checkpoint:

```bash
python pi05/extract_pi05_features.py
python pi05/train_cola_pi05_frozen.py $ARGS --run_dir runs/cola_pi05_frozen
PYTHONPATH=~/mujoco312 python pi05/eval_cola_pi05_frozen.py --model runs/cola_pi05_frozen/checkpoints/best_model.pkl \
    --episodes 150 --max-control-steps 400 --results-path results/pi05/cola_frozen.json
```

**Decentralised pi0.5 baseline.** One LoRA-finetuned pi0.5 policy per arm, each
from its own wrist camera, with no channel. Train arm B the same way
(`--arm b --run-dir runs/pi05_arm_b`):

```bash
python pi05/train_cola_pi05_lora.py --arm a --no-messages --run-dir runs/pi05_arm_a
PYTHONPATH=~/mujoco312 python pi05/eval_pi05_decentralised.py --run-a runs/pi05_arm_a --run-b runs/pi05_arm_b \
    --episodes 150 --max-control-steps 400 --results-path results/pi05/decentralised.json
```

**CoLA on LoRA-finetuned pi0.5.** The message channel inside pi0.5, both arms
trained jointly (`--no-messages` gives the control):

```bash
python pi05/train_cola_pi05_lora.py --run-dir runs/cola_pi05_lora
PYTHONPATH=~/mujoco312 python pi05/eval_cola_pi05_lora.py --run-dir runs/cola_pi05_lora \
    --episodes 150 --max-control-steps 400 --results-path results/pi05/cola_lora.json
```

LoRA training can run in time-limited segments (`--time-limit-min`); rerunning
the same command resumes where the last segment stopped. The pi0.5 and Octo
marker evaluators can be split into shards with `--seed-offset` and pooled with
`--merge shard*.json --results-path merged.json`.

## License

The ALOHA robot model and meshes in `envs/` are derived from MuJoCo Menagerie
and are distributed under the license in `envs/LICENSE`.
