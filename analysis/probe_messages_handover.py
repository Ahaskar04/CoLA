#!/usr/bin/env python3
"""What does CoLA's message encode on the PLAIN HANDOVER task?

The marker task has a categorical answer -- one of three colours -- and a
logistic probe reads it out of msg_a at 73.3% over 150 episodes against a
matched no-message control at exactly 33.3%. Plain handover has no such label:
nothing discrete distinguishes one episode from another.

What it does have is arm A's PRESENTATION POSE. cola_eval_aloha.py and the
collection policy draw it per episode as

    offset = [0, uniform(-HO_Y_RANGE, +HO_Y_RANGE),      # +/- 10 cm lateral
                 uniform(-HO_Z_RANGE, +HO_Z_RANGE)]      # +/- 6 cm vertical

with x deliberately fixed -- the arms always meet at the same point along the
axis between them. So (y, z) is the ONLY thing that varies between episodes,
and it is therefore the only thing a message could be carrying about the task.

This probe regresses (y, z) out of msg_a and reports R^2 and an RMSE in
centimetres, which is interpretable: an RMSE well under the 10 cm / 6 cm draw
ranges means the message locates A's gripper.

WHY IT MATTERS. Messages are worth ~0.1 points on this task (90.6 +/- 1.7 with,
90.5 without) yet the failure profiles invert almost completely -- with
messages B mostly never reaches the box, without them it reaches but cannot
complete the transfer -- and severing them costs 67% on validation loss in the
3-arm case. Something is being transmitted and used by the loss without
changing the outcome. If the answer is "A's end-effector pose", the null has a
mechanism: B can SEE that once the box enters its wrist view, so the message is
redundant with vision here, while on the hidden-marker task the relevant
variable is invisible to B and the same channel is worth 48 points over chance.

LABEL PROVENANCE. The h5 files do not record the offset as an attribute, so it
is read from box_pos[0] -- the box's position at t=0, while A is still holding
it at the presentation pose. Verified against the draw ranges: y spans
-0.107..+0.066 and z spans 0.208..0.320 over the first six training episodes,
consistent with HO_Y_RANGE=0.10 and HO_Z_RANGE=0.06, and x is constant at
-0.28 as the design intends.

THREE PROBES, as in the marker script:
  self_a  -- the encoder's INPUT. The ceiling: a pose the encoder never saw
             cannot be transmitted.
  msg_a   -- the message. The number of interest.
  msg_a from the NO-MESSAGE checkpoint -- the control. Its encoder still trains
             but its output is zeroed before the decoder reads it, so it has no
             gradient pressure to encode anything. On the marker task this
             control sits at exactly chance; here it should sit at R^2 ~ 0.

Grouped 5-fold CV over all episodes, folds split BY EPISODE. Frames within an
episode share a label and are highly correlated, so a frame-level split would
let the probe memorise episodes and report a meaningless R^2 near 1.

CPU only, no rollouts, no SU.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = "/scratch/users/ntu/ahaskar0/v1/cola-research-code"
sys.path.insert(0, f"{REPO}/handover/cola")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                      / "handover_2arm_marker"))

from probe_messages import build_messages, load_split, normalise_states  # noqa: E402

CACHE = "/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/cache_handover_only_v1"
FEATS = "/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/features_handover_only_v1"
EXP = f"{REPO}/handover/cola/experiments"
MSG_CKPT = f"{EXP}/run_handover_only_unet_wristmsg/checkpoints/best_model.pkl"
NOMSG_CKPT = f"{EXP}/run_handover_only_unet_wristnomsg/checkpoints/best_model.pkl"


def episode_offsets(cache_dir, split):
    """A's presentation (y, z) per episode, from box_pos at t=0."""
    import h5py
    manifest = json.load(open(pathlib.Path(cache_dir) / "split_manifest.json"))
    out = []
    for p in manifest["splits"][split]:
        with h5py.File(p, "r") as f:
            b = f["box_pos"][0]
            out.append((float(b[1]), float(b[2])))
    return np.asarray(out, dtype=np.float32)


def probe_cv(X, Y, ep, n_folds=5, seed=0):
    """Grouped k-fold ridge regression. Returns R^2 per target and RMSE in cm."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=n_folds)
    pred = np.zeros_like(Y)
    for tr, te in gkf.split(X, Y, groups=ep):
        sc = StandardScaler().fit(X[tr])
        # RidgeCV picks alpha per fold by internal CV: msg_a is 64-d against
        # ~18k correlated frames, so an unregularised fit would overfit the
        # frame count rather than the 120 independent episodes.
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), Y[tr])
        pred[te] = m.predict(sc.transform(X[te]))

    # Per-episode: average the frame predictions within an episode, which is the
    # quantity that actually has one label. Frame-level R^2 would be dominated
    # by the longest episodes.
    eps = np.unique(ep)
    yt = np.stack([Y[ep == e][0] for e in eps])
    yp = np.stack([pred[ep == e].mean(0) for e in eps])

    ss_res = ((yt - yp) ** 2).sum(0)
    ss_tot = ((yt - yt.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    rmse_cm = 100.0 * np.sqrt(((yt - yp) ** 2).mean(0))
    return r2, rmse_cm, yt.std(0) * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--feat-dir", default=FEATS)
    ap.add_argument("--msg-ckpt", default=MSG_CKPT)
    ap.add_argument("--nomsg-ckpt", default=NOMSG_CKPT)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print("=" * 74)
    print("PROBE: does the message carry arm A's presentation pose (y, z)?")
    print("=" * 74)

    feats, states, eps, offs, feat_o = [], [], [], [], []
    off_base = 0
    for split in ("train", "val", "test"):
        f, st, ep_idx, lens = load_split(a.cache_dir, a.feat_dir, split)
        o = episode_offsets(a.cache_dir, split)
        if len(o) != len(lens):
            raise SystemExit(f"{split}: {len(o)} episodes but {len(lens)} lens")
        fo = pathlib.Path(a.feat_dir) / f"{split}_features_o.npy"
        feats.append(f)
        states.append(normalise_states(a.cache_dir, st))
        feat_o.append(np.load(fo) if fo.exists() else None)
        # Offset episode ids so a group never spans two splits.
        eps.append(ep_idx + off_base)
        offs.append(o[ep_idx])
        off_base += len(lens)
        print(f"  {split:5s}: {len(lens):3d} episodes, {f.shape[0]:6d} frames")

    X_feat = np.concatenate(feats)
    X_state = np.concatenate(states)
    X_o = np.concatenate(feat_o) if all(x is not None for x in feat_o) else None
    ep = np.concatenate(eps)
    Y = np.concatenate(offs)
    print(f"  POOLED: {off_base} episodes, {X_feat.shape[0]} frames")
    print(f"  target spread (cm): y {100*Y[:,0].std():.1f}  z {100*Y[:,1].std():.1f}")

    results = {"episodes": int(off_base), "probes": {}}

    def run(name, X, note):
        r2, rmse, spread = probe_cv(X, Y, ep, n_folds=a.cv)
        print(f"\n{name}")
        print(f"   dim {X.shape[1]:5d} | R^2  y {r2[0]:+.3f}  z {r2[1]:+.3f}"
              f" | RMSE  y {rmse[0]:.2f} cm  z {rmse[1]:.2f} cm"
              f" | label sd  y {spread[0]:.2f}  z {spread[1]:.2f} cm")
        print(f"   {note}")
        results["probes"][name] = {
            "dim": int(X.shape[1]), "r2_y": float(r2[0]), "r2_z": float(r2[1]),
            "rmse_y_cm": float(rmse[0]), "rmse_z_cm": float(rmse[1]),
        }

    print("\n--- messages-ON checkpoint ---")
    self_a, msg_a, meta = build_messages(a.msg_ckpt, X_feat, X_state, X_o)
    print(f"   {meta}")
    results["msg_ckpt_meta"] = meta
    run("self_a  (encoder INPUT, the ceiling)", self_a,
        "R^2 ~ 0 here -> A's own representation does not encode where it is "
        "presenting, and nothing downstream could transmit it")
    run("msg_a   (the message)", msg_a,
        "the number of interest: does what B receives locate A's gripper?")

    if a.nomsg_ckpt and pathlib.Path(a.nomsg_ckpt).exists():
        print("\n--- no-message checkpoint (CONTROL) ---")
        _, nmsg, nmeta = build_messages(a.nomsg_ckpt, X_feat, X_state, X_o)
        print(f"   {nmeta}")
        if nmeta.get("use_messages") is not False:
            print("   WARNING: this checkpoint does not record use_messages=False")
        results["nomsg_ckpt_meta"] = nmeta
        run("msg_a   (no-message model's encoder)", nmsg,
            "R^2 well above 0 here -> the pose is readable from the features "
            "regardless of the channel, and the msg_a number is not evidence "
            "the channel transmits it on purpose")

    print("\n" + "=" * 74)
    print("Messages are worth ~0.1 points on this task (90.6 +/- 1.7 with,")
    print("90.5 without) while the failure profiles invert. If the message")
    print("carries A's pose, the null has a mechanism: B can SEE that once the")
    print("box enters its wrist view, so the channel is redundant with vision")
    print("here -- and not on the hidden-marker task, where the relevant")
    print("variable never enters B's view at all.")
    print("=" * 74)

    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
