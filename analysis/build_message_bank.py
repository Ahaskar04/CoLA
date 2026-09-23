#!/usr/bin/env python3
"""Record msg_a trajectories per marker colour, for the swap intervention.

For every cached demo episode this replays arm A's stored Octo features and
proprioception through the trained encoder and saves the resulting msg_a
sequence, grouped by the episode's marker colour:

    bank[colour] -> list of (T, d_m) arrays, one per episode of that colour

cola_eval_marker.py --swap-messages then feeds B a donor drawn from a colour
OTHER than the one actually on the box, and the tray B chooses says whether the
message causally drives the routing decision.

WHY RECORDED AND NOT LIVE. A live donor would need arm A re-run on a
counterfactual observation every control step -- the same scene with a
different marker -- which means rendering a second rollout in parallel. A
recorded donor is one array lookup. The cost is that the donor was produced in
a different physical state, so as the rollout diverges the message describes a
scene that is not happening. That weakens a NEGATIVE result (B ignoring the
message could mean B rejects an incoherent one) but not a POSITIVE one: if B
follows the swapped colour's tray anyway, the channel is carrying the colour.

The features are the SAME cached arrays the probe reads, so this measures the
encoder on exactly the inputs it was trained on -- no rendering, no GPU.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_messages import (                                    # noqa: E402
    CACHE, COLORS, FEATS, MSG_CKPT, build_messages, episode_colors,
    load_split, normalise_states)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--feat-dir", default=FEATS)
    ap.add_argument("--ckpt", default=MSG_CKPT)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="all three by default: more donors per colour means a "
                         "swap can draw a fresh one each episode instead of "
                         "reusing a handful")
    ap.add_argument("--out", required=True, help=".npz to write")
    a = ap.parse_args()

    bank = {c: [] for c in COLORS}
    meta = None

    for split in a.splits:
        feat, state, ep_idx, lens = load_split(a.cache_dir, a.feat_dir, split)
        cols = episode_colors(a.cache_dir, split)
        if len(cols) != len(lens):
            raise SystemExit(f"{split}: {len(cols)} episodes in the manifest but "
                             f"{len(lens)} in episode_lens")
        fo = pathlib.Path(a.feat_dir) / f"{split}_features_o.npy"
        _, msg, meta = build_messages(
            a.ckpt, feat, normalise_states(a.cache_dir, state),
            np.load(fo) if fo.exists() else None)

        # Cut the flat (frames, d_m) array back into per-episode trajectories.
        # ep_idx was built by np.repeat over episode_lens, so the boundaries are
        # exactly the cumulative sums.
        bounds = np.concatenate([[0], np.cumsum(lens)])
        for e, colour in enumerate(cols):
            bank[colour].append(msg[bounds[e]:bounds[e + 1]])
        print(f"  {split:5s}: {len(lens):3d} episodes -> "
              f"{ {c: sum(1 for x in cols if x == c) for c in COLORS} }")

    if meta is None:
        raise SystemExit("no splits loaded")
    if meta.get("use_messages") is False:
        raise SystemExit("ABORT: donors built from a SEVERED checkpoint. Its "
                         "encoder never had gradient pressure to encode "
                         "anything -- the probe scores it at exactly chance. "
                         "Use the messages-ON checkpoint.")

    out = {}
    for c in COLORS:
        if not bank[c]:
            raise SystemExit(f"no donor episodes for colour {c!r}")
        # Ragged: episodes differ in length, so store each separately and keep
        # an index rather than padding (padding would feed B zeros at the tail,
        # which is the no-message condition, not a swap).
        out[f"{c}_n"] = np.int32(len(bank[c]))
        for i, arr in enumerate(bank[c]):
            out[f"{c}_{i}"] = arr.astype(np.float32)

    d_m = bank[COLORS[0]][0].shape[-1]
    out["message_dim"] = np.int32(d_m)
    out["colors"] = np.array(COLORS)
    out["checkpoint"] = np.array(str(a.ckpt))

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)

    print(f"\nwrote {a.out}")
    print(f"  message_dim {d_m}")
    for c in COLORS:
        lens_c = [len(x) for x in bank[c]]
        print(f"  {c:7s} {len(bank[c]):3d} donors, "
              f"{min(lens_c)}-{max(lens_c)} steps each")
    print(f"  checkpoint use_messages={meta.get('use_messages')} "
          f"epoch={meta.get('epoch')}")


if __name__ == "__main__":
    main()
