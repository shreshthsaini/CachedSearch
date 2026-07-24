"""E3 verifier rescoring: score a directory of saved mp4s with multiple verifiers.

Verifiers
---------
  imagereward : ImageReward-v1.0 averaged over K uniform frames (same verifier the
                gate used; wraps videogen1.gen.FrameRewardScorer).
  vqascore    : t2v-metrics CLIP-FlanT5-XXL VQAScore averaged over K uniform
                frames (K=8 by default, matching ImageReward).
  videoscore  : TIGER-Lab/VideoScore-v1.1 (Mantis-8B-Idefics2 regression head, 5
                aspect scores in [1,4]). Loads with PLAIN transformers==4.49 via a
                ~40-line vendored Idefics2ForSequenceClassification (verbatim logic
                from TIGER-AI-Lab/Mantis modeling_idefics2.py: linear `score` head on
                the last non-pad token) -- no mantis-vl install, no dep conflicts.
  (VisionReward-Video was evaluated and REJECTED for this stack: 13B CogVLM2-Video,
   trust_remote_code custom stack needing decord/pytorchvideo (no aarch64 wheels),
   and many checklist-question forward passes per video. VideoScore is one
   forward pass on a plain-transformers architecture.)

Ensemble = mean of z-scored verifier scores (z over the whole scored set; z-scoring
is per-key affine, so per-prompt rankings of each verifier are unchanged -- it only
sets the combination weights).

WHY `index` MODE EXISTS (read this before touching filenames)
-------------------------------------------------------------
b1_gate.py / b1_temporal.py name videos `{abs(hash(prompt))%1e8}_....mp4`.
PYTHONHASHSEED is UNSET in the generation jobs, so Python randomizes str hashes
per process: the prompt->hash mapping is unrecoverable arithmetic-ally, differs
across shards, and even changes if a shard resumes. `index` rebuilds the mapping
by CLIP retrieval: embed all candidate prompts, embed a few frames from each
hash-group's videos, assign each group the argmax-similarity prompt. Margins are
logged; low-margin groups are flagged for manual review.

Usage (E3 over E2's saved videos)
---------------------------------
  # 1) build video->prompt manifest (one GPU, minutes; resumable):
  python experiments/b1_rescore.py index \
      --videos-dir $CACHEDSEARCH_RESULTS/b1_gate_vbench/videos \
      --prompts experiments/prompts_vbench946.txt --tag vbench

  # 2) score, sharded + resumable (skips files already scored with all verifiers):
  python experiments/b1_rescore.py score \
      --videos-dir $CACHEDSEARCH_RESULTS/b1_gate_vbench/videos --tag vbench \
      --verifiers imagereward,videoscore --shard 0 --num-shards 8

  # 3) fuse + write combined.jsonl with z-scores and ensemble:
  python experiments/b1_rescore.py ensemble --tag vbench

Output jsonl per video: {path, file, hash, prompt, seed/variant (or tag/arm),
index_sim, index_margin, imagereward, vqascore, videoscore_visual,
videoscore_temporal, videoscore_dynamic, videoscore_alignment,
videoscore_factual, videoscore_avg}.
E1 temporal dir works identically with --prompts experiments/your_prompts.txt.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
SCRATCH_HF = os.environ.get("HF_HOME", "./hf_cache")
VIDEOSCORE_ID = os.environ.get("VIDEOSCORE_ID", "TIGER-Lab/VideoScore-v1.1")

# ---------------------------------------------------------------- file helpers

def list_videos(videos_dir):
    vids = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not vids:
        sys.exit(f"no .mp4 files in {videos_dir}")
    return vids


def hash_of(path):
    return os.path.basename(path).split("_")[0]


def parse_name(path):
    """{hash}_{seed}_{variant}.mp4 (b1_gate) or {hash}_{tau_tag}_{keep|commit}.mp4
    (b1_temporal). Returns a dict of whatever fields apply."""
    toks = os.path.splitext(os.path.basename(path))[0].split("_")
    out = {"hash": toks[0]}
    if len(toks) == 3 and toks[1].isdigit():
        out["seed"], out["variant"] = int(toks[1]), toks[2]
    elif len(toks) >= 3:
        out["tau_tag"], out["arm"] = toks[1], toks[2]
    return out


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def out_dir(args):
    return args.out_dir or os.path.join(RESULTS, f"b1_verifiers_{args.tag}")

# ---------------------------------------------------------------- index mode

def cmd_index(args):
    import torch, clip  # openai CLIP (git install, already in videogen1 env)
    from PIL import Image
    from videogen1.video_io import read_video_np

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    odir = out_dir(args)
    manifest = os.path.join(odir, "manifest.jsonl")
    done = {r["hash"] for r in read_jsonl(manifest)}

    groups = {}
    for v in list_videos(args.videos_dir):
        groups.setdefault(hash_of(v), []).append(v)
    todo = {h: vs for h, vs in groups.items() if h not in done}
    print(f"{len(groups)} hash groups, {len(todo)} to index, {len(prompts)} prompts")
    if args.unique and done and todo:
        sys.exit("--unique needs a single-pass global assignment: delete "
                 f"{manifest} and re-run (partial manifests can't be extended 1-1)")
    if args.unique and len(groups) > len(prompts):
        sys.exit(f"--unique impossible: {len(groups)} groups > {len(prompts)} prompts "
                 "(shard restarts created duplicate hash groups; run without --unique)")
    if not todo:
        open(os.path.join(odir, "manifest.done"), "w").write("ok\n")
        return

    model, preprocess = clip.load(
        "ViT-B/32", device=device,
        download_root=os.path.join(SCRATCH_HF, "clip"))
    with torch.no_grad():
        tfeats = []
        for i in range(0, len(prompts), 256):
            t = clip.tokenize(prompts[i:i + 256], truncate=True).to(device)
            f = model.encode_text(t).float()
            tfeats.append(f / f.norm(dim=-1, keepdim=True))
        tfeats = torch.cat(tfeats)  # [P, D]

    hashes, sim_rows = [], []
    for h, vids in sorted(todo.items()):
        imgs = []
        for v in vids[: args.index_videos]:
            frames = read_video_np(v)
            idx = np.linspace(0, len(frames) - 1, args.index_frames).astype(int)
            imgs += [preprocess(Image.fromarray(frames[i])) for i in idx]
        with torch.no_grad():
            f = model.encode_image(torch.stack(imgs).to(device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            sims = (f @ tfeats.T).mean(0).cpu().numpy()  # [P]
        hashes.append(h)
        sim_rows.append(sims)
        print(f"[{h}] embedded ({len(vids)} videos)")

    S = np.stack(sim_rows)  # [G, P]
    if args.unique:
        # Groups <-> prompts is (near-)bijective when generation had no shard
        # restarts: globally optimal 1-1 assignment resolves near-duplicate
        # prompts that per-group argmax collides on (e.g. the two jellyfish
        # prompts in your_prompts.txt). Requires #groups <= #prompts.
        from scipy.optimize import linear_sum_assignment
        gi, pi = linear_sum_assignment(-S)
        assign = dict(zip(gi.tolist(), pi.tolist()))
    else:
        assign = {g: int(S[g].argmax()) for g in range(len(hashes))}

    low = 0
    for g, h in enumerate(hashes):
        p_idx = assign[g]
        order = np.argsort(S[g])[::-1]
        runner = int(order[1] if order[0] == p_idx else order[0])
        margin = float(S[g, p_idx] - S[g, runner])
        rec = dict(hash=h, prompt=prompts[p_idx], prompt_idx=p_idx,
                   sim=float(S[g, p_idx]), margin=margin,
                   argmax_agrees=bool(order[0] == p_idx),
                   runner_up=prompts[runner][:60],
                   n_videos=len(groups[h]))
        append_jsonl(manifest, rec)
        low += margin < args.margin_warn
        print(f"[{h}] sim={rec['sim']:.3f} margin={margin:+.3f} "
              f"{'' if rec['argmax_agrees'] else '(reassigned by --unique) '}"
              f":: {rec['prompt'][:60]}")
    if low:
        print(f"WARNING: {low} groups with margin < {args.margin_warn} "
              f"-- review those rows in {manifest} before trusting E3 on them")
    open(os.path.join(odir, "manifest.done"), "w").write("ok\n")
    print("manifest done:", manifest)

# ---------------------------------------------------------------- verifiers

class ImageRewardVerifier:
    keys = ("imagereward",)

    def __init__(self, device, num_frames=8):
        from videogen1.gen import FrameRewardScorer  # read-only reuse of shared lib
        self.scorer = FrameRewardScorer(k=num_frames, device=device)

    def score(self, frames, prompt):
        return {"imagereward": self.scorer.score(frames, prompt)}


class VQAScoreVerifier:
    keys = ("vqascore",)

    def __init__(self, device, num_frames=8):
        from PIL import Image
        from t2v_metrics import VQAScore

        self.num_frames = num_frames
        self.model = VQAScore(
            model="clip-flant5-xxl", device=device, cache_dir=SCRATCH_HF)

        # t2v-metrics normally accepts image paths. Keep its public scoring API
        # while allowing already-decoded frames, so rescoring does not write
        # thousands of temporary JPEGs.
        path_loader = self.model.model.image_loader

        def load_frame(image):
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
            return path_loader(image)

        self.model.model.image_loader = load_frame

    def score(self, frames, prompt):
        idx = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
        sampled = [frames[i] for i in idx]
        scores = self.model(images=sampled, texts=[prompt])
        return {"vqascore": float(scores.detach().float().mean().cpu())}


# Prompt template verbatim from the VideoScore-v1.1 model card.
_VIDEOSCORE_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge
for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8
For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


class VideoScoreVerifier:
    keys = ("videoscore_visual", "videoscore_temporal", "videoscore_dynamic",
            "videoscore_alignment", "videoscore_factual", "videoscore_avg")

    def __init__(self, device, num_frames=24):
        import torch
        import torch.nn as nn
        # videogen1 env ships timm 0.6.x (image-reward dep); transformers 4.49's
        # AutoModel mapping scan imports timm_wrapper which wants timm>=1.0 attrs.
        # Shim the two names instead of upgrading timm in the SHARED env (running
        # generation arrays import timm via ImageReward's BLIP).
        import timm.data
        for name in ("ImageNetInfo", "infer_imagenet_subset"):
            if not hasattr(timm.data, name):
                setattr(timm.data, name, type(name, (), {}))
        from transformers import AutoProcessor
        from transformers.models.idefics2.modeling_idefics2 import (
            Idefics2Model, Idefics2PreTrainedModel)

        class Idefics2ForSequenceClassification(Idefics2PreTrainedModel):
            """Minimal vendored copy of Mantis' class (TIGER-AI-Lab/Mantis,
            mantis/models/idefics2/modeling_idefics2.py): Idefics2Model backbone +
            `score` linear head, pooled at the last non-pad token. Checkpoint keys
            (model.*, score.*) load unchanged into plain transformers Idefics2."""

            def __init__(self, config):
                super().__init__(config)
                self.num_labels = config.num_labels
                self.model = Idefics2Model(config)
                self.score = nn.Linear(config.text_config.hidden_size, self.num_labels)
                self.post_init()

            def tie_weights(self):  # no output embeddings in this head
                return

            def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                        pixel_attention_mask=None):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 pixel_values=pixel_values,
                                 pixel_attention_mask=pixel_attention_mask,
                                 return_dict=True)
                logits = self.score(out.last_hidden_state)
                bs = input_ids.shape[0]
                if self.config.pad_token_id is None:
                    seq_idx = torch.full((bs,), -1, device=logits.device)
                else:  # last non-pad token (Mantis pooling, ONNX-safe modulo form)
                    seq_idx = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    seq_idx = (seq_idx % input_ids.shape[-1]).to(logits.device)
                return logits[torch.arange(bs, device=logits.device), seq_idx]

        self.torch = torch
        self.num_frames = num_frames
        self.device = device
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        if not str(device).startswith("cuda"):
            print("WARNING: VideoScore is 8B -- CPU/f32 needs ~34GB RAM; login "
                  "nodes (8GB vmem) WILL OOM. Use a GPU node.", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(VIDEOSCORE_ID)
        self.model = Idefics2ForSequenceClassification.from_pretrained(
            VIDEOSCORE_ID, torch_dtype=dtype).eval().to(device)

    def score(self, frames, prompt):
        from PIL import Image
        torch = self.torch
        idx = np.linspace(0, len(frames) - 1, min(self.num_frames, len(frames))).astype(int)
        imgs = [Image.fromarray(frames[i]) for i in idx]
        text = _VIDEOSCORE_PROMPT.format(text_prompt=prompt)
        text += "<image> " * (len(imgs) - text.count("<image>"))
        inputs = self.processor(text=text, images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).float().cpu().numpy()[0]  # [5]
        d = {k: round(float(v), 4) for k, v in zip(self.keys[:5], logits)}
        d["videoscore_avg"] = round(float(np.mean(logits)), 4)
        return d


VERIFIERS = {
    "imagereward": ImageRewardVerifier,
    "vqascore": VQAScoreVerifier,
    "videoscore": VideoScoreVerifier,
}

# ---------------------------------------------------------------- score mode

def cmd_score(args):
    import torch
    from videogen1.video_io import read_video_np

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    odir = out_dir(args)
    manifest = {r["hash"]: r for r in read_jsonl(os.path.join(odir, "manifest.jsonl"))}
    if not manifest:
        sys.exit(f"no manifest at {odir}/manifest.jsonl -- run `index` first "
                 "(filename hashes are NOT recomputable, see module docstring)")

    names = [n.strip() for n in args.verifiers.split(",")]
    for n in names:
        if n not in VERIFIERS:
            sys.exit(f"unknown verifier {n!r}; have {list(VERIFIERS)}")

    vids = list_videos(args.videos_dir)[args.shard::args.num_shards]
    log = os.path.join(odir, f"scores_shard{args.shard}.jsonl")
    prev = {}
    for r in read_jsonl(log):          # resumable; last line per file wins
        prev[r["file"]] = r

    scorers = {}  # lazy: only load models that are actually needed
    n_done = 0
    for path in vids:
        f = os.path.basename(path)
        h = hash_of(path)
        if h not in manifest:
            print(f"SKIP {f}: hash {h} not in manifest", file=sys.stderr)
            continue
        rec = dict(prev.get(f) or {})
        missing = [n for n in names if VERIFIERS[n].keys[0] not in rec]
        if not missing:
            n_done += 1
            continue
        m = manifest[h]
        rec.update(path=os.path.abspath(path), file=f, prompt=m["prompt"],
                   index_sim=m["sim"], index_margin=m["margin"], **parse_name(path))
        frames = read_video_np(path)
        for n in missing:
            if n not in scorers:
                num_frames = {
                    "imagereward": args.ir_frames,
                    "vqascore": args.vqa_frames,
                    "videoscore": args.vs_frames,
                }[n]
                kw = {"num_frames": num_frames}
                scorers[n] = VERIFIERS[n](device=device, **kw)
            rec.update(scorers[n].score(frames, m["prompt"]))
        append_jsonl(log, rec)
        shown = {k: v for k, v in rec.items()
                 if k in ("imagereward", "vqascore", "videoscore_avg")}
        print(f"[{f}] {shown} :: {m['prompt'][:50]}")
    print(f"shard {args.shard}: {n_done} already done, log: {log}")

# ---------------------------------------------------------------- ensemble mode

def cmd_ensemble(args):
    odir = out_dir(args)
    recs = {}
    for f in sorted(glob.glob(os.path.join(odir, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            recs[r["file"]] = {**recs.get(r["file"], {}), **r}
    recs = list(recs.values())
    if not recs:
        sys.exit(f"no scores_shard*.jsonl under {odir}")

    keys = [k.strip() for k in args.ensemble_keys.split(",")]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in recs if k in r], dtype=float)
        if len(vals) < 2:
            sys.exit(f"verifier key {k!r} has {len(vals)} values -- score it first")
        stats[k] = (vals.mean(), vals.std() + 1e-8)
        print(f"{k}: n={len(vals)} mean={vals.mean():.4f} std={vals.std():.4f}")

    combined = os.path.join(odir, "combined.jsonl")
    open(combined, "w").close()
    n_full = 0
    for r in recs:
        zs = []
        for k in keys:
            if k in r:
                z = (r[k] - stats[k][0]) / stats[k][1]
                r[f"z_{k}"] = round(float(z), 4)
                zs.append(z)
        if len(zs) == len(keys):
            r["ensemble"] = round(float(np.mean(zs)), 4)
            n_full += 1
        append_jsonl(combined, r)
    print(f"wrote {combined}: {len(recs)} videos, {n_full} with full ensemble")

    both = [r for r in recs if all(k in r for k in keys)]
    if len(both) > 10 and len(keys) > 1:
        from scipy.stats import spearmanr
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                rho = spearmanr([r[keys[i]] for r in both],
                                [r[keys[j]] for r in both]).statistic
                print(f"global spearman({keys[i]}, {keys[j]}) = {rho:.3f} "
                      f"(per-prompt rank agreement is E3's job -- see b1_analyze)")

# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tag", default="vbench")
    common.add_argument("--out-dir", default=None,
                        help="default: $CACHEDSEARCH_RESULTS/b1_verifiers_<tag>")
    common.add_argument("--device", default=None, help="cuda|cpu (default: auto)")

    p = sub.add_parser("index", parents=[common],
                       help="build video->prompt manifest via CLIP retrieval")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--prompts", required=True,
                   help="candidate prompt list (experiments/prompts_vbench946.txt "
                        "for E2, experiments/your_prompts.txt for E1)")
    p.add_argument("--index-videos", type=int, default=4,
                   help="videos sampled per hash group")
    p.add_argument("--index-frames", type=int, default=4, help="frames per video")
    p.add_argument("--margin-warn", type=float, default=0.01)
    p.add_argument("--unique", action="store_true",
                   help="globally optimal 1-1 group<->prompt assignment (Hungarian);"
                        " use when each prompt was generated by exactly one shard"
                        " process (no restarts), e.g. E1 b1_temporal")

    p = sub.add_parser("score", parents=[common], help="score videos (sharded)")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--verifiers", default="imagereward,videoscore")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--ir-frames", type=int, default=8,
                   help="ImageReward frames (8 = what the gate used)")
    p.add_argument("--vqa-frames", type=int, default=8,
                   help="VQAScore frames (8 = ImageReward sampling protocol)")
    p.add_argument("--vs-frames", type=int, default=24,
                   help="VideoScore frames (card supports up to 48)")

    p = sub.add_parser("ensemble", parents=[common],
                       help="z-score + mean -> combined.jsonl")
    p.add_argument("--ensemble-keys", default="imagereward,videoscore_avg")

    args = ap.parse_args()
    {"index": cmd_index, "score": cmd_score, "ensemble": cmd_ensemble}[args.mode](args)


if __name__ == "__main__":
    main()
