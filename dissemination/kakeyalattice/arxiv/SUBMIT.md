# arXiv submission walkthrough — KakeyaLattice

Est. time: **10 minutes** of active work + endorsement wait (hours to days
if first-time cs.LG submitter, instant if already endorsed).

## Prerequisites

- An arXiv account (register at https://arxiv.org/user/register)
- cs.LG endorsement (if first-time — see `endorsement_request.md`)
- LaTeX toolchain (`pdflatex`, `bibtex`) — optional but recommended

## Step 1 — Build the submission tarball

From the KakeyaLattice repo root:

```bash
bash dissemination/kakeyalattice/arxiv/build_tarball.sh
```

Output: `dissemination/kakeyalattice/arxiv/arxiv_submission.tar.gz`

Sanity-check:

```bash
tar -tzf dissemination/kakeyalattice/arxiv/arxiv_submission.tar.gz | head -20
```

You should see `kakeyalattice.tex` and (if pdflatex was available) a
pre-built `kakeyalattice.bbl`.

## Step 2 — Fill the submission form

Go to https://arxiv.org/submit → "Start a new submission".

Paste values from `metadata.yaml`:

| Form field | Value source |
|---|---|
| Title | `title` |
| Author(s) | `authors` (single author: Allen Li) |
| Abstract | `abstract` (paste as-is; arXiv strips LaTeX automatically) |
| Comments | `comments` |
| Primary subject | **cs.LG** |
| Cross-listing | cs.CL, cs.IT, cs.DS |
| MSC class | 94A29, 68T50 |
| ACM class | I.2.7; E.4 |
| License | **CC BY 4.0** (recommended) |

## Step 3 — Upload tarball

- Choose "Upload: tar archive of sources"
- Upload `arxiv_submission.tar.gz`
- Wait for server-side build (typical: 2–5 min)
- If build fails: the error log usually points to a missing figure or package;
  copy it into the tarball and rebuild.

## Step 4 — Preview PDF

arXiv auto-generates a preview PDF. Compare against the source PDF at
`reports/paper/kakeyalattice.pdf`; they should be visually identical. If the
preview is missing references or figures, fix the tarball and resubmit.

## Step 5 — Submit

Click "Submit" on the metadata page. You'll get an immediate confirmation
with a temporary ID (like `submit/12345678`). The permanent
`arXiv:26MM.NNNNN` ID is assigned after the next daily announcement cycle
(Monday–Thursday announce at 20:00 UTC; Friday's submissions announce Monday).

## Step 6 — After publication

Once you have the arXiv ID, update KakeyaLattice in this order:

```bash
# In FluffyAIcode/LLM-KV--Cache-compress:

# 6a. Badge + citation in README
# Add to top of README.md:
#   [![arXiv](https://img.shields.io/badge/arXiv-26MM.NNNNN-b31b1b.svg)](https://arxiv.org/abs/26MM.NNNNN)

# 6b. Update Papers with Code entry (see ../paperswithcode/)
# 6c. Update HF Space README badge (see ../huggingface/space/README.md)
# 6d. Post the arXiv link as a comment on the vLLM issue (see ../vllm_issue/)
# 6e. Reply to NestQuant / NexusQuant threads with the arXiv link for reverse backlinks
```

Google Scholar usually indexes within **24–48 h** of arXiv publication.
Semantic Scholar and Connected Papers within **1–3 days**.

## Common pitfalls

- **Non-ASCII characters** in the abstract field: replace em-dashes (—) with
  double-hyphens (--), and curly quotes with straight quotes. metadata.yaml
  already does this.
- **Missing `.bbl`**: if arXiv can't find your bibliography, either
  pre-build it (the script does this when pdflatex is available) or include
  the `.bib` file and ensure `\bibliography{kakeyalattice}` points to it.
- **Figures > 6 MB**: compress PDFs with `gs -sDEVICE=pdfwrite
  -dPDFSETTINGS=/ebook`.
- **Version update**: if you revise the paper post-publication (v1.5 adds
  new data, for example), submit as a **replacement** from the same abstract
  page, not as a new submission. Each version gets `v1`, `v2` suffixes under
  the same arXiv ID.
