# Papers with Code submission walkthrough

Est. time: **3 minutes** (do this *after* arXiv is live — you'll paste the
arXiv ID into the form).

## Prerequisites

- Papers with Code account (free, https://paperswithcode.com/accounts/login)
- An arXiv ID (ideally) or a public PDF URL (fine; the repo PDF at
  `reports/paper/kakeyalattice.pdf` works)

## Step 1 — Submit the paper

Go to https://paperswithcode.com/paper/submit

Paste fields from `entry.json`:

| Form field | Source in `entry.json` |
|---|---|
| Title | `paper.title` |
| Authors | `paper.authors` (one per line) |
| Abstract | `paper.abstract_short` |
| arXiv link | `paper.arxiv_id` → `https://arxiv.org/abs/<id>` |
| PDF URL | `paper.pdf_url` (fallback if arXiv not live yet) |
| Published date | `paper.published_date` |

PwC will fetch the abstract from arXiv if the ID is given; the text in
`entry.json` is the fallback.

## Step 2 — Link the code

On the paper page, click **"Add Code"**:

| Field | Value |
|---|---|
| Repository URL | `https://github.com/FluffyAIcode/LLM-KV--Cache-compress` |
| Framework | PyTorch |
| Is official? | ✅ yes |
| Mentioned in paper? | ✅ yes |

## Step 3 — Tag tasks and methods

PwC's taxonomy is hierarchical. Apply:

**Tasks** (from `entry.json.tasks`):
- Language Modelling
- Quantization
- Model Compression
- Efficient Transformers

**Methods** (from `entry.json.methods`):
- Vector Quantization
- (create new if not listed) Nested Lattice Quantization
- (create new if not listed) E8 Lattice
- Hadamard Transform

PwC lets you create new methods if they don't exist. "Nested Lattice
Quantization" and "E8 Lattice" currently don't have method pages —
creating them (even with minimal descriptions) gives KakeyaLattice a
permanent backlink from every future paper that adopts either method.

## Step 4 — Add leaderboard rows (optional but high-value)

PwC leaderboards are what drives traffic. For each row in
`entry.json.leaderboard_rows`:

1. Find the matching benchmark page (e.g. "KV Cache Compression on
   WikiText-103"). If none exists, click **"Add Benchmark"** under
   Tasks → Quantization. Name it using the `benchmark` field.
2. Click **"Add Result"**:
   - Method name: `KakeyaLattice v1.5 (E8)` or `KakeyaLattice v1.4 (D4)`
   - Paper: the paper page you just created
   - Model: the HF model ID (copy from `models_evaluated`)
   - Metric values: from the row
   - Extra info: hardware + protocol string

Leaderboard rows are the #1 driver of long-tail PwC traffic to a paper.

## Step 5 — Link the HF Space (after you deploy it)

PwC paper pages have a "Spaces" section that pulls from the HF hub if
the Space's `paper` tag matches the arXiv ID. Ensure the Space's
`README.md` YAML frontmatter has:

```yaml
paper: 26MM.NNNNN
```

(Fill in after arXiv is live.) This links the Space to the paper on both
sides automatically.

## Step 6 — Sanity check

- The paper page at `https://paperswithcode.com/paper/kakeyalattice-...`
  should now show: code link, arXiv link, abstract, ≥1 leaderboard row.
- Google typically indexes PwC paper pages within 24–48 h.
- PwC's own search is instant — your paper should be findable by title or
  by any of the tagged methods/tasks immediately after submission.
