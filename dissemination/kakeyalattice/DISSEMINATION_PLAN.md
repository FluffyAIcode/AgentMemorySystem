# KakeyaLattice Dissemination Kit

**Target repo**: [`FluffyAIcode/LLM-KV--Cache-compress`](https://github.com/FluffyAIcode/LLM-KV--Cache-compress)
**Goal**: move the project from "搜不到" (discoverable only by exact name) to "natural-language discoverable" on the four primary channels researchers / engineers actually use.

## Why this kit exists

KakeyaLattice v1.4/v1.5 is fully measured and release-ready, but as of 2026-04-27 generic queries like *"lattice KV cache compression"*, *"E8 KV quant vLLM"*, or *"Kakeya-Zamir LLM"* return NexusQuant / NestQuant / KV-Compress / LMCache — not this repo. The four causes we can actually fix from the author side:

1. GitHub repo has **zero topics** → excluded from `/topics/*` discovery pages.
2. No arXiv ID → no Google Scholar / Semantic Scholar / Connected Papers index → no academic backlinks.
3. No vLLM-ecosystem issue → not cross-referenced from the 76k-star vLLM repo (NexusQuant got this via `vllm#39241` and it's already its #1 inbound source).
4. No HuggingFace Space and no Papers with Code entry → no `paperswithcode.com/paper/...` landing page and no HF hub search hit.

This kit completes **what can be automated** (config files, LaTeX tarball builder, issue Markdown, Space scaffold, PwC JSON) and stages **what requires a human account** (arXiv endorsement + upload, HF CLI login + push, PwC submit button) as one-command steps.

## Execution order (5 steps, ~30–40 min of human time total)

| # | Task | Where it lives | Who executes | Time |
|---|------|----------------|--------------|------|
| 1 | Set GitHub topics + description | `github_topics/apply.sh` | repo owner, 1 command | 30 s |
| 2 | Submit arXiv preprint | `arxiv/` | Allen Li, arXiv account | 10 min (+ endorsement wait) |
| 3 | Open vLLM Discussion / Issue | `vllm_issue/BODY.md` | anyone with GitHub account | 2 min |
| 4 | Deploy HuggingFace Space demo | `huggingface/space/` | any HF account | 5 min |
| 5 | Submit Papers with Code entry | `paperswithcode/` | any PwC account | 3 min |

After all five land, you should have **4 new inbound backlinks** (vLLM issue, HF Space, arXiv abstract page, PwC paper page) and **7 GitHub topic pages** pointing at the repo. Empirically this is the minimum needed to show up on natural-language LLM + KV-cache queries.

## Per-step quick start

```bash
# 1. GitHub topics (run from any machine with gh CLI auth'd as repo owner)
bash dissemination/kakeyalattice/github_topics/apply.sh

# 2. Build arXiv tarball (produces arxiv_submission.tar.gz, upload at arxiv.org/submit)
bash dissemination/kakeyalattice/arxiv/build_tarball.sh
# Then follow dissemination/kakeyalattice/arxiv/SUBMIT.md

# 3. Open vLLM issue (body ready at vllm_issue/BODY.md)
gh issue create -R vllm-project/vllm \
    --title "$(cat dissemination/kakeyalattice/vllm_issue/TITLE.txt)" \
    --body-file dissemination/kakeyalattice/vllm_issue/BODY.md

# 4. Deploy HF Space
bash dissemination/kakeyalattice/huggingface/deploy.sh   # requires `huggingface-cli login`

# 5. Submit to Papers with Code (manual, 30 seconds) — see paperswithcode/SUBMIT.md
```

## Files in this kit

```
dissemination/kakeyalattice/
├── DISSEMINATION_PLAN.md        ← this file
├── github_topics/
│   ├── topics.json              ← topic list (source of truth)
│   ├── description.txt          ← GitHub "About" one-liner
│   └── apply.sh                 ← `gh` CLI command, one-shot
├── arxiv/
│   ├── SUBMIT.md                ← submission walkthrough (endorsement, categories)
│   ├── metadata.yaml            ← title, authors, abstract, categories, comment
│   ├── build_tarball.sh         ← produces arxiv_submission.tar.gz from reports/paper/
│   └── endorsement_request.md   ← template email to request cs.LG endorsement
├── vllm_issue/
│   ├── TITLE.txt                ← issue title
│   ├── BODY.md                  ← issue body (mirrors NexusQuant vllm#39241 format)
│   └── LABELS.txt               ← recommended labels
├── huggingface/
│   ├── space/                   ← full HF Space repo scaffold (app.py, requirements.txt, README.md)
│   ├── deploy.sh                ← pushes Space to hf.co/spaces/<user>/kakeyalattice
│   └── MODEL_CARD_EDIT.md       ← snippet to add to any HF model card that benefits from KakeyaLattice
└── paperswithcode/
    ├── SUBMIT.md                ← submit walkthrough
    ├── entry.json               ← paper + code + results (copy-paste ready)
    └── sota_tables.md           ← pre-filled iso-PPL and iso-bit leaderboard rows
```

## Measurement of success

After execution, re-run these natural-language queries; each should surface the repo or its arXiv page in the first result page (currently zero do):

- `lattice KV cache compression vLLM`
- `E8 lattice KV cache quantization`
- `Kakeya-Zamir nested lattice LLM`
- `D4 E8 KV cache H200`
- `KV cache compression plugin vLLM 2026`

We expect first Google indexing of the arXiv page within **24–72 h** and first Bing/DuckDuckGo within **5–7 days** post-submission. GitHub topics update is immediate. HF Space and PwC typically index within 24 h.
