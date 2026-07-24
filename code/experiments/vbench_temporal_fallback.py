"""VBench temporal-subset metrics, raw reimplementation (fallback / cross-check).

Runs in the MAIN videogen1 env (no vbench install needed). Use when the isolated
VBench env is not built yet, or as an independent cross-check of official numbers.

Metrics (per video, all in [0,1] except flow magnitude):
  subject_consistency    DINO ViT-S/16 frame-feature similarity, VBench formula:
                         mean_t 0.5*(cos(f_1,f_t)+cos(f_{t-1},f_t)), clamped >= 0.
  background_consistency same formula with CLIP ViT-B/32 features (VBench formula).
  motion_smoothness      RAFT flow-compensated warping error between consecutive
                         frames: 1 - MAE(warp(f_{t+1} <- flow), f_t)/255.
                         PROXY: official VBench uses AMT-S frame interpolation, so
                         absolute values are NOT comparable to VBench leaderboard
                         numbers -- but keep-vs-commit paired DELTAS (E1) are valid.
  temporal_flickering    1 - mean_t MAE(f_t, f_{t+1})/255. Official VBench applies
                         this only to static-scene prompts; on arbitrary videos it
                         mixes motion into the signal -- again, use for paired deltas.
  dynamic_degree_flow    mean RAFT flow magnitude (px, at native resolution scale)
                         over consecutive pairs. Raw value, not VBench's binary.

Sharded + resumable (same pattern as b1_gate.py). Example (E1 keep-vs-commit):
  python experiments/vbench_temporal_fallback.py \
      --videos-dir $CACHEDSEARCH_RESULTS/b1_temporal/videos --tag b1_temporal \
      --shard 0 --num-shards 4
Output: $CACHEDSEARCH_RESULTS/vbench_temporal_fallback_<tag>/scores_shard{i}.jsonl
        rows {path, file, ...parsed name fields..., metric: value}.
Model caches go to scratch (TORCH_HOME / CLIP download_root), never $HOME.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
SCRATCH_HF = os.environ.get("HF_HOME", "./hf_cache")
# torch.hub (DINO) + torchvision (RAFT) weights: keep off $HOME (quota!)
os.environ.setdefault("TORCH_HOME", os.path.join(SCRATCH_HF, "torch"))

ALL_METRICS = ("subject_consistency", "background_consistency",
               "motion_smoothness", "temporal_flickering", "dynamic_degree_flow")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def parse_name(path):
    toks = os.path.splitext(os.path.basename(path))[0].split("_")
    out = {"hash": toks[0]}
    if len(toks) == 3 and toks[1].isdigit():
        out["seed"], out["variant"] = int(toks[1]), toks[2]
    elif len(toks) >= 3:
        out["tau_tag"], out["arm"] = toks[1], toks[2]
    return out


# ------------------------------------------------------------------ models

class Models:
    """Lazy per-metric model loading so cheap metrics never pull heavy weights."""

    def __init__(self, device, flow_scale=0.5, raft="large", batch=8):
        self.device, self.flow_scale, self.raft_kind, self.batch = device, flow_scale, raft, batch
        self._dino = self._clip = self._raft = None

    def dino(self):
        if self._dino is None:
            import torch
            self._dino = torch.hub.load("facebookresearch/dino:main",
                                        f"dino_vits16").eval().to(self.device)
        return self._dino

    def clip(self):
        if self._clip is None:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device,
                                          download_root=os.path.join(SCRATCH_HF, "clip"))
            self._clip = (model.eval(), preprocess)
        return self._clip

    def raft(self):
        if self._raft is None:
            from torchvision.models.optical_flow import (
                raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights)
            if self.raft_kind == "large":
                w = Raft_Large_Weights.DEFAULT
                m = raft_large(weights=w)
            else:
                w = Raft_Small_Weights.DEFAULT
                m = raft_small(weights=w)
            self._raft = (m.eval().to(self.device), w.transforms())
        return self._raft


def _feat_consistency(feats):
    """VBench formula: mean over t>=1 of 0.5*(cos(f_1,f_t) + cos(f_{t-1},f_t)),
    cosines clamped at 0. feats: [N, D] L2-normalized torch tensor."""
    first = (feats[1:] @ feats[0]).clamp(min=0)
    consec = (feats[1:] * feats[:-1]).sum(-1).clamp(min=0)
    return float(((first + consec) / 2).mean())


def subject_consistency(frames, M):
    import torch
    import torchvision.transforms.functional as TF
    model = M.dino()
    imgs = []
    for f in frames:
        t = torch.from_numpy(f).permute(2, 0, 1).float() / 255
        t = TF.resize(t, 224, antialias=True)
        t = TF.center_crop(t, 224)
        imgs.append(TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), M.batch):
            f = model(torch.stack(imgs[i:i + M.batch]).to(M.device)).float()
            feats.append(f / f.norm(dim=-1, keepdim=True))
    return {"subject_consistency": round(_feat_consistency(torch.cat(feats)), 4)}


def background_consistency(frames, M):
    import torch
    from PIL import Image
    model, preprocess = M.clip()
    imgs = [preprocess(Image.fromarray(f)) for f in frames]
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), M.batch):
            f = model.encode_image(torch.stack(imgs[i:i + M.batch]).to(M.device)).float()
            feats.append(f / f.norm(dim=-1, keepdim=True))
    return {"background_consistency": round(_feat_consistency(torch.cat(feats)), 4)}


def temporal_flickering(frames, M):
    a = frames.astype(np.float32)
    mae = np.abs(a[1:] - a[:-1]).mean()
    return {"temporal_flickering": round(float(1 - mae / 255), 4)}


def _flow_pairs(frames, M):
    """RAFT flow for consecutive pairs at flow_scale, /8-aligned. Yields
    (flow[B,2,h,w], src[B,3,h,w], dst[B,3,h,w]) float tensors in [0,1]."""
    import torch
    import torchvision.transforms.functional as TF
    model, transforms = M.raft()
    H, W = frames.shape[1:3]
    h = max(128, int(H * M.flow_scale) // 8 * 8)  # RAFT needs >=128px sides
    w = max(128, int(W * M.flow_scale) // 8 * 8)
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    t = TF.resize(t, [h, w], antialias=True)
    for i in range(0, len(t) - 1, M.batch):
        src = t[i:i + M.batch]          # frames t
        dst = t[i + 1:i + 1 + M.batch]  # frames t+1
        n = min(len(src), len(dst))
        src, dst = src[:n], dst[:n]
        a, b = transforms(src.to(M.device), dst.to(M.device))
        with torch.no_grad():
            flow = model(a, b)[-1]      # [n,2,h,w], px displacement src->dst
        yield flow, src.to(M.device), dst.to(M.device)


def motion_smoothness(frames, M):
    """1 - warping error: backward-warp f_{t+1} to t using RAFT flow, MAE vs f_t."""
    import torch
    errs = []
    for flow, src, dst in _flow_pairs(frames, M):
        n, _, h, w = flow.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=flow.device),
                                torch.arange(w, device=flow.device), indexing="ij")
        base = torch.stack([xx, yy]).float()[None]          # [1,2,h,w]
        pos = base + flow                                    # sample dst at src+flow
        grid = torch.stack([pos[:, 0] / (w - 1) * 2 - 1,
                            pos[:, 1] / (h - 1) * 2 - 1], dim=-1)  # [n,h,w,2]
        warped = torch.nn.functional.grid_sample(
            dst, grid, mode="bilinear", padding_mode="border", align_corners=True)
        errs.append(torch.abs(warped - src).mean().item() * 255)
    return {"motion_smoothness": round(float(1 - np.mean(errs) / 255), 4)}


def dynamic_degree_flow(frames, M):
    import torch
    mags = []
    for flow, _, _ in _flow_pairs(frames, M):
        mags.append(torch.linalg.vector_norm(flow, dim=1).mean().item() / M.flow_scale)
    return {"dynamic_degree_flow": round(float(np.mean(mags)), 4)}


METRIC_FNS = {
    "subject_consistency": subject_consistency,
    "background_consistency": background_consistency,
    "motion_smoothness": motion_smoothness,
    "temporal_flickering": temporal_flickering,
    "dynamic_degree_flow": dynamic_degree_flow,
}

# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--tag", default=None, help="default: basename of parent dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--metrics", default=",".join(ALL_METRICS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="uniformly subsample to N frames (0 = all; use ~9 for CPU tests)")
    ap.add_argument("--flow-scale", type=float, default=0.5)
    ap.add_argument("--raft", choices=("large", "small"), default="large")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from videogen1.video_io import read_video_np
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    metrics = [m.strip() for m in args.metrics.split(",")]
    for m in metrics:
        if m not in METRIC_FNS:
            sys.exit(f"unknown metric {m!r}; have {list(METRIC_FNS)}")

    tag = args.tag or os.path.basename(os.path.dirname(
        os.path.join(args.videos_dir, "")))
    odir = args.out_dir or os.path.join(RESULTS, f"vbench_temporal_fallback_{tag}")
    log = os.path.join(odir, f"scores_shard{args.shard}.jsonl")
    prev = {}
    for r in read_jsonl(log):
        prev[r["file"]] = r

    vids = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    if not vids:
        sys.exit(f"no .mp4 in {args.videos_dir}")
    vids = vids[args.shard::args.num_shards]

    M = Models(device, flow_scale=args.flow_scale, raft=args.raft, batch=args.batch)
    for path in vids:
        f = os.path.basename(path)
        rec = dict(prev.get(f) or {})
        missing = [m for m in metrics if m not in rec]
        if not missing:
            continue
        rec.update(path=os.path.abspath(path), file=f, **parse_name(path))
        frames = read_video_np(path)
        if args.max_frames and len(frames) > args.max_frames:
            frames = frames[np.linspace(0, len(frames) - 1, args.max_frames).astype(int)]
        for m in missing:
            rec.update(METRIC_FNS[m](frames, M))
        append_jsonl(log, rec)
        print(f"[{f}] " + " ".join(f"{m}={rec[m]}" for m in metrics))
    print("shard done:", log)


if __name__ == "__main__":
    main()
