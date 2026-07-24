"""Calibrate the caching threshold for a model we did not test.

    python examples/calibrate_new_model.py --model <hf-id> --prompts data/calib25.txt

Costs about two GPU-hours for a small model (25 prompts x 8 seeds, twice).
Fidelity tracks architecture family, not parameter size, so calibrate once per
family and reuse the threshold across sizes within it.
"""
import argparse

import torch
from diffusers import DiffusionPipeline

from cachedsearch import calibrate_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--num-prompts", type=int, default=25)
    ap.add_argument("--taus", default="0.02,0.05,0.10,0.20")
    ap.add_argument("--target-capture", type=float, default=0.90)
    args = ap.parse_args()

    prompts = [l.strip() for l in open(args.prompts) if l.strip()][: args.num_prompts]
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    report = calibrate_tau(
        pipe,
        prompts,
        taus=[float(t) for t in args.taus.split(",")],
        target_capture=args.target_capture,
    )

    print(f"{'tau':>6} {'speedup':>9} {'capture':>9} {'median rho':>11}")
    for tau, row in report.items():
        if tau == "recommended":
            continue
        print(f"{tau:>6} {row['speedup']:>8.2f}x {100*row['capture']:>8.1f}% {row['spearman']:>11.3f}")
    print(f"\nrecommended tau for this model: {report['recommended']}")
    print("Use it as: cached_search(pipe, prompt, verifier, tau=<value>)")


if __name__ == "__main__":
    main()
