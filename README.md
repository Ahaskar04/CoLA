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
preprocess/        demo -> training cache, and frozen Octo-Base feature extraction
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

Main environment (CoLA, Octo baselines, data collection), Python 3.10 or 3.11 with a
CUDA GPU:

```bash
git clone https://github.com/octo-models/octo.git
pip install -e octo
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install numpy==1.24.3        # last: jaxlib 0.4.20 needs the numpy 1.x ABI
```

For headless rendering set `export MUJOCO_GL=egl`. All commands below run from
the repository root.

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
# Two-arm handover
python preprocess/prepare_cache_2arm.py
python preprocess/prepare_states_2arm.py
python preprocess/extract_features_2arm.py --overhead

# Hidden-marker handover (same scripts, its own folders)
python preprocess/prepare_cache_2arm.py --data_dir data/demos/handover_marker --cache_dir data/cache/handover_marker
python preprocess/prepare_states_2arm.py --cache_dir data/cache/handover_marker
python preprocess/extract_features_2arm.py --cache_dir data/cache/handover_marker --feat_dir data/features/handover_marker

# Three-arm handover: 130 episodes, sampled with seed 42
python preprocess/prepare_cache_3arm.py --max_episodes 130 --seed 42
python preprocess/extract_features_3arm.py --overhead
```

Splits are 80/10/10 by episode. `--overhead` also extracts the shared overhead
camera, which the overhead-only runs below use.

### 3. Train CoLA

Every run uses the same objective and budget:

```bash
ARGS="--lambda_gripper 0.25 --grip_transition_weight 0 --diffusion --diffusion_unet --num_epochs 150"
```

Each configuration has a no-message control (`--no_messages`), identical
except that the channel is severed.

```bash
# Two-arm: overhead camera only, and wrist cameras only
python train/train_2arm.py $ARGS --overhead_only --run_dir runs/handover_2arm
python train/train_2arm.py $ARGS --overhead_only --no_messages --run_dir runs/handover_2arm_nomsg
python train/train_2arm.py $ARGS --run_dir runs/handover_2arm_wrist
python train/train_2arm.py $ARGS --no_messages --run_dir runs/handover_2arm_wrist_nomsg

# Hidden marker: wrist cameras only
M="--cache_dir data/cache/handover_marker --feat_dir data/features/handover_marker"
python train/train_2arm.py $ARGS $M --run_dir runs/handover_marker
python train/train_2arm.py $ARGS $M --no_messages --run_dir runs/handover_marker_nomsg

# Three-arm: overhead camera only
python train/train_3arm.py $ARGS --overhead_only --run_dir runs/handover_3arm
python train/train_3arm.py $ARGS --overhead_only --no_messages --run_dir runs/handover_3arm_nomsg
```

The best checkpoint (lowest validation loss) is written to
`<run_dir>/checkpoints/best_model.pkl`.

### 4. Evaluate in simulation

Evaluation episodes are seeded by index, so every model sees the same start
states. `--seed-offset` selects a different block; where we report several
blocks we use offsets 0, 1000, 2000, 3000 and 4000. Pass `--no-messages` when
evaluating a `*_nomsg` run (the evaluator refuses a mismatch). `--no-videos`
skips rendering the rollout videos. In the two-arm evaluation, an episode whose
scripted start state loses the grasp is skipped: it is left out of the rates
and counted under `skipped` in the results.

```bash
# Two-arm: 200 episodes per block
for s in 0 1000 2000 3000 4000; do
  python eval/eval_2arm.py --model runs/handover_2arm/checkpoints/best_model.pkl \
      --handover-only --episodes 200 --seed-offset $s \
      --results-path results/handover_2arm/seed$s.json
done

# Hidden marker: 201 episodes (67 per colour)
python eval/eval_marker.py --model runs/handover_marker/checkpoints/best_model.pkl \
    --episodes 201 --results-path results/handover_marker/results.json

# Three-arm: 200 episodes, success = C holds the box
python eval/eval_3arm.py --model runs/handover_3arm/checkpoints/best_model.pkl \
    --episodes 200 --results-path results/handover_3arm/results.json
```

Each evaluator prints a summary and writes it, with per-episode records, to the
results JSON.

### 5. Hidden-marker analyses

```bash
MODEL=runs/handover_marker/checkpoints/best_model.pkl

# Linear probes on the messages (CPU only): marker colour, and, on the plain
# two-arm handover (wrist-only runs), A's presentation offset
python analysis/probe_messages_marker.py
python analysis/probe_messages_2arm.py

# Message swap: B receives A's message from an episode with a different colour
python analysis/build_message_bank.py --out data/marker_message_bank.npz
python eval/eval_marker.py --model $MODEL --episodes 201 \
    --swap-messages data/marker_message_bank.npz --results-path results/handover_marker/swap.json

# Out-of-distribution scenes: box colour and lighting
for v in box_orange box_grey box_white light_bright light_dim light_dark; do
  python eval/eval_marker.py --model $MODEL --episodes 201 \
      --scene-xml envs/handover_marker/scene_$v.xml --results-path results/handover_marker/ood_$v.json
done

# Message size: train with --message_dim 16, 32 or 128 (64 is the default), then evaluate as above
python train/train_2arm.py $ARGS $M --message_dim 16 --run_dir runs/handover_marker_d16

# Channel transfer: reuse the two-arm channel frozen, or freeze a random channel; only the heads train
python train/train_2arm.py $ARGS $M --freeze_channel \
    --init_channel_from runs/handover_2arm_wrist/checkpoints/best_model.pkl --run_dir runs/handover_marker_xfer
python train/train_2arm.py $ARGS $M --freeze_channel --run_dir runs/handover_marker_frozen_random
```

### 6. Octo baselines

Octo-Base finetuned per arm with no channel between arms (decentralised), or
as a single policy that controls both arms (centralised, `--arm both`). All
rows use a U-Net action head, a batch of 128 and the same number of gradient
steps as the matching CoLA run. `--tune head_only` keeps the backbone frozen,
as in CoLA; without it the whole model is finetuned.

```bash
OCTO="--pretrained-path hf://rail-berkeley/octo-base --head unet --action-horizon 12 --batch-size 128 --grad-accum 2"

# Two-arm (12,600 steps). Overhead camera shown; use --image-key image_wrist_a / image_wrist_b for the wrist rows.
HO="--manifest data/cache/handover_2arm/split_manifest.json --steps 12600 --image-key image_overhead"
python baselines/octo/finetune_octo.py $OCTO $HO --arm a    --save-dir runs/octo/ho_a
python baselines/octo/finetune_octo.py $OCTO $HO --arm b    --save-dir runs/octo/ho_b
python baselines/octo/finetune_octo.py $OCTO $HO --arm both --save-dir runs/octo/ho_central

# Hidden marker (20,250 steps; each arm uses its own wrist camera)
MK="--manifest data/cache/handover_marker/split_manifest.json --steps 20250"
python baselines/octo/finetune_octo.py $OCTO $MK --arm a --save-dir runs/octo/marker_a
python baselines/octo/finetune_octo.py $OCTO $MK --arm b --save-dir runs/octo/marker_b

# Three-arm (26,850 steps)
TA="--manifest data/cache/handover_3arm/split_manifest.json --steps 26850 --image-key image_overhead"
for a in a b c; do
  python baselines/octo/finetune_octo.py $OCTO $TA --arm $a --save-dir runs/octo/3arm_$a
done
```

`--resume` continues an interrupted run. Evaluation uses the same scoring code
as CoLA, and every evaluation script checks this at startup.

```bash
# Two-arm, over the same seed blocks as CoLA. Wrist rows add --scene-xml envs/handover_2arm/scene.xml.
python baselines/octo/eval_octo_2arm.py --checkpoint-a runs/octo/ho_a --checkpoint-b runs/octo/ho_b \
    --handover-only --episodes 200 --criterion hold --seed-offset 0 --results-path results/octo/ho_seed0.json
python baselines/octo/eval_octo_2arm.py --checkpoint-a runs/octo/ho_central --centralised \
    --handover-only --episodes 200 --criterion hold --seed-offset 0 --results-path results/octo/ho_central_seed0.json

python baselines/octo/eval_octo_marker.py --checkpoint-a runs/octo/marker_a --checkpoint-b runs/octo/marker_b \
    --episodes 150 --results-path results/octo/marker.json
python baselines/octo/eval_octo_3arm.py --checkpoint-a runs/octo/3arm_a --checkpoint-b runs/octo/3arm_b \
    --checkpoint-c runs/octo/3arm_c --episodes 200 --results-path results/octo/3arm.json

# Zero-shot Octo-Base on the two-arm scene, including the pick-up (--handover-only starts from the handover)
python baselines/octo/eval_octo_zeroshot.py --octo-checkpoint octo-base --episodes 200
```

### 7. pi0.5 backbone study (hidden marker)

These scripts run in a separate environment with
[openpi](https://github.com/Physical-Intelligence/openpi), installed as
described in its README. openpi pins an older MuJoCo, so install MuJoCo 3.12.0
(the version used for every other evaluation) into its own directory and put
it first on `PYTHONPATH` for the evaluation scripts only:

```bash
uv pip install h5py imageio imageio-ffmpeg tqdm
uv pip install --no-deps --target ~/mujoco312 mujoco==3.12.0
```

Every pi0.5 script builds the model from the pi0.5 base checkpoint, read from
openpi's cache (`~/.cache/openpi` unless `OPENPI_DATA_HOME` is set). Download it
once with openpi's own helper:

```bash
python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')"
```

They use the hidden-marker cache from step 2 (`data/cache/handover_marker`).

**CoLA on frozen pi0.5.** CoLA's adapters, channel and U-Net head on pooled
PaliGemma prefix features (2048-d) instead of Octo-Base's:

```bash
python pi05/extract_pi05_features.py
python pi05/train_cola_pi05_frozen.py $ARGS --run_dir runs/cola_pi05_frozen
python pi05/train_cola_pi05_frozen.py $ARGS --no_messages --run_dir runs/cola_pi05_frozen_nomsg
PYTHONPATH=~/mujoco312 python pi05/eval_cola_pi05_frozen.py --model runs/cola_pi05_frozen/checkpoints/best_model.pkl \
    --episodes 150 --max-control-steps 400 --results-path results/pi05/cola_frozen.json
```

**Decentralised pi0.5 baseline.** One LoRA-finetuned pi0.5 policy per arm, each
from its own wrist camera, with no channel:

```bash
python pi05/train_cola_pi05_lora.py --arm a --no-messages --run-dir runs/pi05_arm_a
python pi05/train_cola_pi05_lora.py --arm b --no-messages --run-dir runs/pi05_arm_b
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
