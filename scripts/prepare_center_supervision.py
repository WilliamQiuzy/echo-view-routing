#!/usr/bin/env python
"""Freeze the training windows and initial states of the target-time supervision round
(configs/experiments/ev9v_center_supervision_v1.yaml; plan Task 1).

Checks the pinned inputs (protocol, E1 export, reference embeddings, probe, feature manifest), the cache_v2 frame-order
stamp and that the training bank rebuilt from the protocol keeps its digest. Builds one track per training clip
(cache_v2 frame indices, public labels) and per training stream (the frozen recipes), lists the mixed-view windows of
the training streams, and writes for every seed:
  SC_s{seed}.npz   the windows of S and C (one file, read by both arms)
  N_s{seed}.npz    the windows of N (the same native draws; each mixed window replaced by its target recording's own window)
  init_s{seed}.pt  the initial temporal head and both classifier heads (torch.manual_seed(seed); heads from the E1 probe)
plus index.json: every file's sha256, the per-epoch input digests, mixed-window availability per class and exposure
counts. Only the protocol's training clips enter; no validation label is read.

  python scripts/prepare_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml \
      --out runs/center_supervision_v1/manifests
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.config import load_paths, require_current_cache  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.features.extract import VARIANT_GAMMA  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.dev_run import check_index, digest  # noqa: E402
from echo_routing.vit.target_video import TargetTimeHead  # noqa: E402
from echo_routing.vit.target_windows import (GAMMA, Track, build_manifests, check_train_only, epoch_digest,  # noqa: E402
                                             mixed_index, native_track)

PINNED = ("protocol", "visual_export", "visual_reference", "probe", "feature_manifest")


def check_inputs(cs: dict, root: Path) -> dict:
    out = {}
    for key in PINNED:
        p, want = cs["inputs"][key]["path"], cs["inputs"][key]["sha256"]
        if (got := sha256_file(root / p)) != want:
            raise SystemExit(f"{key} {p}: sha256 {got[:12]} != {want[:12]}")
        out[key] = got
    return out


def tracks(cfg: dict, cs: dict, paths, proto: dict) -> tuple[dict[int, list[Track]], list[Track], set[str]]:
    key = cs["inputs"]["cache_key"]; require_current_cache(paths.cache_root / "features" / key)
    idx = read_index(variant_dir(paths.cache_root, key, "orig")); check_index(idx, proto)
    bank = multi_bank(idx, "train", cfg)
    if digest([recipe_to_dict(r) for r in bank.recipes]) != proto["train"]["multi_bank"]["digest"]:
        raise SystemExit("the rebuilt training bank does not match the protocol digest")
    store = FeatureStore(paths.cache_root, key); labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
    train = list(proto["train"]["clips"]); natives: dict[int, list[Track]] = {k: [] for k in range(len(cs["classes"]))}
    for v in sorted(train):
        natives[labels[v]].append(native_track(v, store.get(v, "orig").frame_idx, labels[v]))
    streams = []
    for r in bank.recipes:
        vids, frames, gammas, labs, clip_idx = [], [], [], [], []
        for fr in r.fragments:
            vf = store.get(fr.video_id, "orig"); n = fr.end - fr.start
            vids += [fr.video_id] * n; frames.append(vf.frame_idx[fr.start:fr.end]); gammas.append(np.full(n, float(VARIANT_GAMMA[fr.variant])))
            labs.append(np.full(n, int(fr.label))); clip_idx.append(np.arange(fr.start, fr.end))
        streams.append(Track(r.recipe_id, tuple(vids), np.concatenate(frames).astype(int), np.concatenate(gammas), np.concatenate(labs),
                             np.concatenate(clip_idx)))
    used = {v for t in streams for v in t.videos} | {t.seq_id for k in natives for t in natives[k]}
    check_train_only(used, set(train))
    return natives, streams, set(train)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs = cfg["center_supervision"]; paths = load_paths(); root = paths.repo_root
    hashes = check_inputs(cs, root); proto = json.loads((root / cs["inputs"]["protocol"]["path"]).read_text())
    if cs["sampling"]["native_gamma"] != {"orig": 0.5, "gamma090": 0.5} or GAMMA != {"orig": 1.0, "gamma090": 0.9}:
        raise SystemExit("native_gamma in the config must match draw_native (orig and gamma090 with probability 0.5 each)")
    natives, streams, train = tracks(cfg, cs, paths, proto)
    mix = mixed_index(streams); names = cs["classes"]
    avail = {names[k]: {"recordings": len(mix.get(k, {})), "windows": int(sum(len(w) for w in mix.get(k, {}).values()))} for k in range(len(names))}
    if any(v["recordings"] == 0 for v in avail.values()):
        raise SystemExit(f"a class has no mixed-view window in the training streams: {avail}")
    sp = cs["sampling"]; n_upd = int(sp["epochs"]) * int(sp["updates_per_epoch"]); upe = int(sp["updates_per_epoch"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probe = torch.load(root / cs["inputs"]["probe"]["path"], map_location="cpu", weights_only=False)["state_dict"]
    m, t = cs["model"], cs["model"]["temporal"]
    index = {"config": a.config, "inputs": hashes, "training_clips": len(train), "training_streams": len(streams),
             "natives_per_class": {names[k]: len(v) for k, v in natives.items()}, "mixed_availability": avail, "seeds": {}}
    for seed in cs["seeds"]:
        sc, n_arm = build_manifests(int(seed), int(sp["data_seed_base"]), natives, streams, mix, n_upd,
                                    int(sp["native_per_update"]), int(sp["mixed_per_update"]))
        files = {}
        for name, arrs in (("SC", sc), ("N", n_arm)):
            path = out / f"{name}_s{seed}.npz"; np.savez(path, **arrs); files[name] = path.name
        torch.manual_seed(int(seed))
        head = TargetTimeHead(512, int(t["layers"]), int(t["heads"]), int(t["ffn"]), float(t["dropout"]), int(cs["window"]["max_offset"]),
                              len(names), float(m["position_init_std"]))
        head.init_from_probe(probe); torch.save(head.state_dict(), out / f"init_s{seed}.pt"); files["init"] = f"init_s{seed}.pt"
        expo = Counter(sc["videos"][sc["target_vid"]].ravel().tolist())
        index["seeds"][str(seed)] = {
            "files": {k: {"file": f, "sha256": sha256_file(out / f)} for k, f in files.items()},
            "epoch_digests": {k: [epoch_digest(arr, e * upe, (e + 1) * upe) for e in range(int(sp["epochs"]))] for k, arr in (("SC", sc), ("N", n_arm))},
            "windows": int(sc["kind"].size), "mixed_windows": int((sc["kind"] == 1).sum()),
            "mixed_by_target_class": {names[k]: int(((sc["kind"] == 1) & (sc["target"] == k)).sum()) for k in range(len(names))},
            "repeated_positions": {"SC": int(sc["repeats"].sum()), "N": int(n_arm["repeats"].sum())},
            "windows_with_repeats": {"SC": int((sc["repeats"] > 0).sum()), "N": int((n_arm["repeats"] > 0).sum())},
            "target_exposure_SC": {"recordings": len(expo), "min": min(expo.values()), "median": float(np.median(list(expo.values()))),
                                   "max": max(expo.values())},
            "training_clips_never_a_target_SC": len(train - set(expo))}
        print(f"seed {seed}: {json.dumps(index['seeds'][str(seed)]['mixed_by_target_class'])}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({k: v for k, v in index.items() if k != "seeds"}, indent=1)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
