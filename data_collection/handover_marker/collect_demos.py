"""Record demonstrations from a scripted expert.

COLA_POLICY selects the expert module (scripted_policy by default). Episodes
whose randomised reset loses the grasp are re-run, so the output holds
NUM_EPISODES demos. Output goes to data/demos/<task> (<task>_fulltask without
COLA_HANDOVER_ONLY=1) unless COLA_OUTPUT_DIR is set.
"""

import os
from importlib import import_module
from pathlib import Path

import h5py
import mediapy as media

REPO = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent.name

# Expert module to record.
POLICY_MODULE = os.environ.get("COLA_POLICY", "scripted_policy")
policy = import_module(POLICY_MODULE)
setup, run_episode = policy.setup, policy.run_episode
box_x_range, box_y_range = policy.box_x_range, policy.box_y_range

# Handover-only mode: every episode starts with arm A already holding the box.
HANDOVER_ONLY = os.environ.get("COLA_HANDOVER_ONLY", "0") == "1"

# Default output directory (override with COLA_OUTPUT_DIR).
_default_out = REPO / "data" / "demos" / (TASK if HANDOVER_ONLY else f"{TASK}_fulltask")
output_dir = Path(os.environ.get("COLA_OUTPUT_DIR", _default_out))
output_dir.mkdir(parents=True, exist_ok=True)

NUM_EPISODES = int(os.environ.get("COLA_NUM_EPISODES", "200"))
MAX_ATTEMPTS = int(os.environ.get("COLA_MAX_ATTEMPTS", str(NUM_EPISODES * 3)))
REVIEW_FPS = int(os.environ.get("COLA_REVIEW_FPS", "50"))

print(f"policy:   {POLICY_MODULE}")
print(f"mode:     {'HANDOVER ONLY (box starts in gripper A)' if HANDOVER_ONLY else 'full task'}")
print(f"output:   {output_dir.resolve()}")
print(f"episodes: {NUM_EPISODES} (max {MAX_ATTEMPTS} attempts)")

kept = 0
attempts = 0
successes = 0
grasp_rejects = 0

while kept < NUM_EPISODES and attempts < MAX_ATTEMPTS:
    attempts += 1
    print(f"Running episode {kept} (attempt {attempts})...")

    _kw = {"handover_only": True, "episode_idx": kept} if HANDOVER_ONLY else {}
    result = run_episode(setup, box_x_range, box_y_range, **_kw)

    # Box dropped during the reset; nothing recorded, so retry.
    if result[0] is None:
        grasp_rejects += 1
        print(f"  DISCARDED: grasp lost during reset ({grasp_rejects} so far)")
        continue

    episode_data, success, review_frames = result

    # Failed episodes are kept too; their action labels are still valid.
    if success:
        successes += 1

    with h5py.File(output_dir / f"episode_{kept:04d}.h5", "w") as f:
        for key, value in episode_data.to_dict().items():
            f.create_dataset(key, data=value)
        f.attrs["success"] = success
        # Provenance.
        f.attrs["policy_module"] = POLICY_MODULE
        f.attrs["handover_only"] = HANDOVER_ONLY
        # Also stored as an attribute, so scoring needn't load the images.
        if episode_data.marker_color is not None:
            f.attrs["marker_color"] = episode_data.marker_color

    media.write_video(str(output_dir / f"episode_{kept:04d}_review.mp4"),
                      review_frames, fps=REVIEW_FPS)

    kept += 1

print(f"\nDone. {kept}/{NUM_EPISODES} episodes written in {attempts} attempts.")
print(f"  task success: {successes}/{kept}")
print(f"  discarded (grasp lost at reset): {grasp_rejects}")

if kept < NUM_EPISODES:
    raise SystemExit(
        f"\nSTOPPED EARLY: only {kept} of {NUM_EPISODES} episodes after "
        f"{attempts} attempts (raise COLA_MAX_ATTEMPTS)."
    )
