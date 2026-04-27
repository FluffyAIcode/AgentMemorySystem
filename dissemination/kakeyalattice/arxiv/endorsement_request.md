# arXiv cs.LG Endorsement Request — Email Template

If you have never submitted to `cs.LG` before, arXiv requires an endorsement
from an existing cs.LG author. Endorsements are **per category**, not per paper.

## How to get the endorsement code

1. Register at https://arxiv.org/user/register
2. Click "Endorse" in the user menu → arXiv generates a 6-character code
   (e.g. `X3K9PZ`) and a 3-digit identifier (e.g. `allen_li_1`)
3. Send the email below to any of the suggested endorsers (they can endorse
   you with one click at `https://arxiv.org/auth/endorse?x=<CODE>`)

## Who to ask (in priority order)

All of these have recent cs.LG papers that KakeyaLattice directly compares
against or builds on:

| Endorser | Affiliation | Relevant work | Contact channel |
|---|---|---|---|
| **Semyon Savkin** | MIT LIDS | NestQuant (nested lattice quantisation, ICML 2025) | `savkin@mit.edu` — most aligned |
| **Yury Polyanskiy** | MIT EECS | NestQuant co-author | arXiv author page |
| **Ram Zamir** | Tel Aviv University | Foundational Zamir–Feder nested lattices cited in the paper | TAU website |
| João Marques | Independent | NexusQuant (E8 KV quant) | via `@jagmarques` on GitHub |
| Isaac Rehg | Independent | KV-Compress (PagedAttention integration) | via `@IsaacRe` on GitHub |

## Email template

```
Subject: arXiv cs.LG endorsement request — KV-cache lattice compression paper

Dear Prof./Dr. <LAST NAME>,

I'm Allen Li, an independent researcher. I have a paper ready for arXiv
submission titled "KakeyaLattice: Nested-Lattice KV-Cache Compression with
Kakeya-Style Discrete Codebooks (D4 + E8 Joint Release)", which directly
extends/compares-against your work on <NestQuant / nested lattices for
matrix products / KV compression>.

The paper constructs a discrete Kakeya cover via a Zamir–Feder nested-lattice
quantiser and demonstrates that the D4 and E8 shaping gains (+0.37 dB and
+0.66 dB over Z^N) materialise in live-vLLM on H200 with +1.3 to +2.0 dB
measured per-layer K-MSE gain. It is fully open-source, Apache-2.0, with
reproducible H200 harnesses at
https://github.com/FluffyAIcode/LLM-KV--Cache-compress

This is my first cs.LG submission, so arXiv requires endorsement. Would you
be willing to endorse me for cs.LG? My arXiv endorsement code is:

    <PASTE 6-CHAR CODE HERE>

The endorsement link is:
    https://arxiv.org/auth/endorse?x=<PASTE CODE HERE>

Happy to share the full PDF upfront — it's at
https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf

Thank you for considering,

Allen Li
AllenL329@gmail.com
```

## After endorsement

Run:

```bash
bash dissemination/kakeyalattice/arxiv/build_tarball.sh
```

then upload `arxiv_submission.tar.gz` at https://arxiv.org/submit with the
fields from `metadata.yaml`.

Expected arXiv ID appearance: **within 24 h of submission**, typically as
`arXiv:26MM.NNNNN` for a late-April 2026 submission.

## Post-submission: update the repo

After you have the arXiv ID, run from the repo root:

```bash
# Replace 26MM.NNNNN with your actual arXiv ID
NEW_ID=26MM.NNNNN
sed -i '' "s|reports/paper/kakeyalattice.pdf|arXiv:$NEW_ID (reports/paper/kakeyalattice.pdf)|g" README.md
```

and add the arXiv badge to `README.md`:

```markdown
[![arXiv](https://img.shields.io/badge/arXiv-26MM.NNNNN-b31b1b.svg)](https://arxiv.org/abs/26MM.NNNNN)
```

This one badge alone is worth ~50% of the search-indexing uplift on Google
Scholar / Semantic Scholar.
