<p align="center">
  <img src="assets/logo.png" alt="CachedSearch" width="720">
</p>

<p align="center">
  Training-free cached exploration for test-time search in video diffusion
</p>

<p align="center">
  Paper (arXiv, coming soon) &nbsp;·&nbsp;
  <a href="https://shreshthsaini.github.io/CachedSearch">Project page</a>
  &nbsp;·&nbsp; Blog (coming soon)
</p>

<p align="center">
  <img src="assets/fig1_overview.png" alt="CachedSearch overview" width="900">
</p>

- Ranking survives caching: median Spearman rho is 0.905, with 72% top-1 agreement on VBench.
- Ranking failures are self-limiting because they concentrate where candidates are nearly tied.
- Explore cheap plus commit full retains 94.7% of the best-of-8 gain at 63% of the cost.
- CachedSearch works with any published cache engine and transfers across six models in four families.

## How it works

<p align="center">
  <img src="assets/fig2_method.png" alt="CachedSearch method" width="900">
</p>

Generate every candidate with a training-free cache enabled.<br>
Score the cached drafts with the verifier used by the search procedure.<br>
Select the seed with the highest draft score.<br>
Regenerate only that seed at full compute.<br>
Return the full-compute video, so caching affects selection but not delivery quality.

```text
drafts = [cached_generate(prompt, seed) for seed in seeds]
winner = seeds[argmax(verifier(video, prompt) for video in drafts)]
return full_generate(prompt, winner)
```

## Results at a glance

Each row uses the listed cache as the exploration engine at search width 8. Cost includes the full-compute commit.

| Exploration engine | Gain capture | Cost |
|---|---:|---:|
| No caching | 100.0% | 621 s |
| PAB | 99.5% | 493 s |
| FasterCache CFG-Cache | 99.6% | 465 s |
| TeaCache | 93.2% | 365 s |
| EasyCache | 90.1% | 346 s |

<p align="center">
  <img src="assets/f16_hero.png" alt="Delivered reward versus search budget" width="850">
</p>

At similar exploration cost, caching retains 90.1% of search gain, while 25-step truncation retains 72.6%.

## Quickstart

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the 50-prompt gate experiment:

```bash
PYTHONPATH=code CACHEDSEARCH_RESULTS=./results python code/experiments/b1_gate.py \
  --prompts code/experiments/prompts_gate50.txt \
  --seeds 8 --steps 50 --tau 0.10 --tag gate50
```

Regenerate the main analysis figures from compatible result shards:

```bash
CACHEDSEARCH_RESULTS=./results CACHEDSEARCH_FIGURES=./assets \
  python code/paper_figs/make_figs.py --out assets
```

A CUDA GPU is required for generation. The reference gate configuration used an NVIDIA GH200 with 96 GB of memory. Smaller GPUs may require model offloading or reduced spatial and temporal resolution.

## Repository layout

```text
.
├── assets/              Figures, logo, and the logo generator
├── code/
│   ├── experiments/     Generation, scoring, and analysis entry points
│   ├── paper_figs/      Scripts that derive figures from result shards
│   └── videogen1/       Cache wrappers and shared video generation utilities
├── data/                The public gate prompt and seed grid
├── LICENSE              MIT license
└── requirements.txt     Unpinned Python dependencies
```

## Data

[`data/prompts_gate50.csv`](data/prompts_gate50.csv) contains all 50 gate prompts crossed with seeds 0 through 7. It has 400 rows with the columns `prompt_id`, `seed`, and `prompt`. Official VBench and VBench-2.0 prompt lists are linked in [`data/README.md`](data/README.md) and are not redistributed.

## Acknowledgments

We sincerely thank the authors and open-source teams whose models, acceleration methods, evaluators, and benchmarks made this work possible: [Wan2.1](https://github.com/Wan-Video/Wan2.1), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [LTX-Video](https://github.com/Lightricks/LTX-Video), [TeaCache](https://github.com/ali-vilab/TeaCache), [PAB and VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [FasterCache](https://github.com/Vchitect/FasterCache), [EasyCache](https://github.com/H-EmbodVis/EasyCache), [ImageReward](https://github.com/THUDM/ImageReward), [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore), and [VBench](https://github.com/Vchitect/VBench). Their public releases made it possible to study cached exploration across a broad and reproducible video generation ecosystem.

## Citation

```bibtex
@article{saini2026cachedsearch,
  title   = {CachedSearch: Training-Free Cached Exploration for Test-Time Search in Video Diffusion},
  author  = {Saini, Shreshth and Birkbeck, Neil and Wang, Yilin and Adsumilli, Balu and Bovik, Alan C.},
  year    = {2026},
  note    = {arXiv id TBD}
}
```

## License

The code and release materials are available under the [MIT License](LICENSE).
