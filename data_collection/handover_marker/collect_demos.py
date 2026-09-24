"""Record demonstrations from a scripted expert.

Which expert is recorded is set by COLA_POLICY. Both the plain and the scanning
variants expose the same four names, so this file does not care which is loaded
-- except in one respect: the scanning expert returns a fourth value,
`scan_found`, and episodes where it is False MUST NOT BE KEPT.

Why: the scanning expert opens with a fixed sweep and reaches for the box only
once its wrist camera has actually seen it. If the sweep runs out before the box
comes into view, the phase machine falls through to `approach` anyway and the arm
flies at a box the camera never found. The recorded action then depends on
information the observation does not contain -- exactly the unlearnable label the
scan phase exists to remove, concentrated in the spawns that are hardest to see.
Keeping those episodes would bias the dataset toward easy spawns AND poison the
approach phase at the same time.

Discarded episodes are re-run with a fresh spawn, so the output always holds
NUM_EPISODES usable demonstrations with contiguous numbering.
"""

import os
from importlib import import_module
from pathlib import Path

import h5py
import mediapy as media

# Which scripted expert to record. Defaults to the plain one, so this behaves
# exactly as before; the scanning collections set
# COLA_POLICY=scripted_policy_scan.
POLICY_MODULE = os.environ.get("COLA_POLICY", "scripted_policy")
policy = import_module(POLICY_MODULE)
setup, run_episode = policy.setup, policy.run_episode
box_x_range, box_y_range = policy.box_x_range, policy.box_y_range

# Handover-only mode: every episode STARTS with arm A already holding the box,
# so the recorded task is the handover alone (move_arm_B onward). Used to
# measure coordination without the grasp -- which accounts for ~80% of failures
# in the full task and otherwise dominates the success rate.
HANDOVER_ONLY = os.environ.get("COLA_HANDOVER_ONLY", "0") == "1"

# $HOME is quota-limited and these datasets run to several GB of images, so
# default to scratch rather than the repo's dataset_cache.
_default_out = ("/scratch/users/ntu/ahaskar0/aloha-handover-only" if HANDOVER_ONLY
                else "/scratch/users/ntu/ahaskar0/aloha-handover-data-v4")
output_dir = Path(os.environ.get("COLA_OUTPUT_DIR", _default_out))
output_dir.mkdir(parents=True, exist_ok=True)

NUM_EPISODES = int(os.environ.get("COLA_NUM_EPISODES", "200"))
MAX_ATTEMPTS = int(os.environ.get("COLA_MAX_ATTEMPTS", str(NUM_EPISODES * 3)))
DISCARD_SCAN_TIMEOUT = os.environ.get("COLA_DISCARD_SCAN_TIMEOUT", "1") == "1"
REVIEW_FPS = int(os.environ.get("COLA_REVIEW_FPS", "50"))

print(f"policy:   {POLICY_MODULE}")
print(f"mode:     {'HANDOVER ONLY (box starts in gripper A)' if HANDOVER_ONLY else 'full task'}")
print(f"output:   {output_dir.resolve()}")
print(f"episodes: {NUM_EPISODES} (max {MAX_ATTEMPTS} attempts)")
print(f"discard timed-out scans: {DISCARD_SCAN_TIMEOUT}")

kept = 0
attempts = 0
successes = 0
scan_timeouts = 0
grasp_rejects = 0

while kept < NUM_EPISODES and attempts < MAX_ATTEMPTS:
    attempts += 1
    print(f"Running episode {kept} (attempt {attempts})...")

    _kw = {"handover_only": True, "episode_idx": kept} if HANDOVER_ONLY else {}
    result = run_episode(setup, box_x_range, box_y_range, **_kw)

    # handover_only returns (None, False, None) when the randomised reset shook
    # the box out of the gripper. Nothing was recorded; retry with a new offset.
    if result[0] is None:
        grasp_rejects += 1
        print(f"  DISCARDED: grasp lost during reset ({grasp_rejects} so far)")
        continue

    # The plain expert returns three values, the scanning one four. Unpack
    # either, and treat a missing scan_found as "no scan phase, nothing to
    # discard".
    if len(result) == 4:
        episode_data, success, review_frames, scan_found = result
    else:
        episode_data, success, review_frames = result
        scan_found = True

    if DISCARD_SCAN_TIMEOUT and not scan_found:
        scan_timeouts += 1
        print(f"  DISCARDED: the sweep timed out without finding the box. "
              f"({scan_timeouts} so far)")
        continue

    # A FAILED episode is still kept: the expert's actions are explainable from
    # its observations throughout, so the labels are valid even though the task
    # did not complete. Only an unobservable label is disqualifying.
    if success:
        successes += 1

    with h5py.File(output_dir / f"episode_{kept:04d}.h5", "w") as f:
        for key, value in episode_data.to_dict().items():
            f.create_dataset(key, data=value)
        f.attrs["success"] = success
        # Provenance, so a cache built from this dataset can be traced back to
        # the expert that produced it.
        f.attrs["scan_found"] = scan_found
        f.attrs["policy_module"] = POLICY_MODULE
        f.attrs["handover_only"] = HANDOVER_ONLY
        # Also an attribute, not just the dataset to_dict() writes: scoring
        # correct-tray placement needs the label, and reading it from attrs
        # avoids opening the image arrays. None in full-task mode, which draws
        # no marker.
        if episode_data.marker_color is not None:
            f.attrs["marker_color"] = episode_data.marker_color

    media.write_video(str(output_dir / f"episode_{kept:04d}_review.mp4"),
                      review_frames, fps=REVIEW_FPS)

    kept += 1

print(f"\nDone. {kept}/{NUM_EPISODES} episodes written in {attempts} attempts.")
print(f"  task success: {successes}/{kept}")
print(f"  discarded (scan timed out): {scan_timeouts}")
print(f"  discarded (grasp lost at reset): {grasp_rejects}")

if kept < NUM_EPISODES:
    raise SystemExit(
        f"\nSTOPPED EARLY: only {kept} of {NUM_EPISODES} episodes after "
        f"{attempts} attempts.\n"
        f"A high discard rate means the sweep cannot see part of the spawn "
        f"range. Run scan_coverage_check() in {POLICY_MODULE} and widen "
        f"SCAN_ANGLE_START/END, loosen SCAN_DETECT_CONE, or move SCAN_VANTAGE "
        f"before collecting again."
    )

if scan_timeouts > 0.1 * attempts:
    print(f"\nWARNING: {100.0 * scan_timeouts / attempts:.0f}% of attempts were "
          f"discarded. The kept episodes are biased toward the spawns the sweep "
          f"happens to cover. Check scan_coverage_check() before training on this.")