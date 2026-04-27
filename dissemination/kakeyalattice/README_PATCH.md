# README patch for FluffyAIcode/LLM-KV--Cache-compress

Paste this block directly below the first heading (`# KakeyaLattice — v1.4
KV-Cache Compression`) in the KakeyaLattice repo's `README.md`. Every
badge is self-updating: they reflect live status as soon as the
corresponding step in the dissemination kit is completed.

```markdown
[![Release v1.5](https://img.shields.io/github/v/release/FluffyAIcode/LLM-KV--Cache-compress?color=blue&label=release)](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/latest)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](reports/paper/kakeyalattice.pdf)
[![Papers with Code](https://img.shields.io/badge/Papers_with_Code-pending-21cbce.svg)](https://paperswithcode.com/paper/kakeyalattice)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-demo-yellow.svg)](https://huggingface.co/spaces/FluffyAIcode/kakeyalattice)
[![vLLM Issue](https://img.shields.io/badge/vLLM-feature_request-informational.svg)](https://github.com/vllm-project/vllm/issues?q=KakeyaLattice)

**Topics**: `kv-cache` · `kv-cache-compression` · `quantization` · `vllm` ·
`lattice-quantization` · `e8-lattice` · `d4-lattice` · `nested-lattice` ·
`llm-inference` · `long-context` · `h200`
```

After arXiv lands, replace the `arXiv-pending` badge line with:

```markdown
[![arXiv](https://img.shields.io/badge/arXiv-26MM.NNNNN-b31b1b.svg)](https://arxiv.org/abs/26MM.NNNNN)
```

and add a **Citation** section at the bottom of `README.md`:

```markdown
## Citation

If you use KakeyaLattice in your research, please cite:

​```bibtex
@misc{li2026kakeyalattice,
  author       = {Allen Li},
  title        = {{KakeyaLattice}: Nested-Lattice {KV}-Cache Compression
                  with {K}akeya-Style Discrete Codebooks ({D}4 + {E}8 Joint Release)},
  year         = {2026},
  eprint       = {26MM.NNNNN},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/26MM.NNNNN},
  note         = {Code: \url{https://github.com/FluffyAIcode/LLM-KV--Cache-compress}}
}
​```
```

## One-command re-dissemination

Add this section somewhere near the end of `README.md`:

```markdown
## Dissemination

To keep the project discoverable (GitHub topics, arXiv, vLLM issue, HF
Space, Papers with Code), use the dissemination kit shipped in
[`dissemination/`](dissemination/DISSEMINATION_PLAN.md).  All five
channels are scripted to one command each:

​```bash
# 1. GitHub topics + description (requires repo-admin gh CLI auth)
bash dissemination/github_topics/apply.sh

# 2. arXiv submission tarball (upload at https://arxiv.org/submit)
bash dissemination/arxiv/build_tarball.sh

# 3. Open a vLLM issue (body pre-written)
gh issue create -R vllm-project/vllm \
    --title "$(cat dissemination/vllm_issue/TITLE.txt)" \
    --body-file dissemination/vllm_issue/BODY.md

# 4. Deploy HF Space (requires huggingface-cli login)
bash dissemination/huggingface/deploy.sh

# 5. Submit to Papers with Code (manual form, 3 min)
#    entries ready at dissemination/paperswithcode/entry.json
```

## Where to drop the kit

The kit currently lives in the `AgentMemorySystem` repo (branch
`AgentMemory/kakeyalattice-dissemination-kit-f31f`). To adopt it into
KakeyaLattice:

```bash
cd LLM-KV--Cache-compress
git remote add ams https://github.com/FluffyAIcode/AgentMemorySystem
git fetch ams AgentMemory/kakeyalattice-dissemination-kit-f31f
git checkout ams/AgentMemory/kakeyalattice-dissemination-kit-f31f -- \
    dissemination/kakeyalattice
git mv dissemination/kakeyalattice dissemination
git commit -m "Adopt KakeyaLattice dissemination kit"
git push
```

From then on, all five steps are re-runnable from inside the KakeyaLattice
repo with no re-staging.
