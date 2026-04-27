# How to open the vLLM issue

Est. time: **2 minutes**.

## Option A — GitHub CLI (recommended)

From any machine with `gh` authenticated:

```bash
gh issue create -R vllm-project/vllm \
    --title "$(cat dissemination/kakeyalattice/vllm_issue/TITLE.txt)" \
    --body-file dissemination/kakeyalattice/vllm_issue/BODY.md
```

`gh` prints the issue URL. Paste it into:

- KakeyaLattice `README.md` ("Integration" section)
- HF Space `README.md` (Resources)
- Papers with Code entry (code_links)

## Option B — Web UI

1. Go to https://github.com/vllm-project/vllm/issues/new/choose
2. Pick the **Feature Request** template
3. Title: copy from `TITLE.txt`
4. Body: copy from `BODY.md`
5. Submit

## After opening

- Don't ping individual maintainers in the issue body. They are watched by
  the `[kv-cache]` and `[performance]` triage rotations and will route it.
- If nobody responds within 7 days, add a polite bump comment linking to
  the arXiv ID (by then hopefully available).
- If a maintainer expresses interest, open a **draft PR** wiring the plugin
  into vLLM's plugin test matrix. That is the fastest route to being listed
  in the vLLM README's "Speculative decoding / KV compression" bullet list,
  which is the single highest-value backlink in this ecosystem.

## Cross-posting (optional)

Consider also posting a summary (with a link back to the vLLM issue) in:

- vLLM Slack `#general` or `#kv-cache` channels
- SGLang Discord (KakeyaLattice already has an SGLang-shaped codec surface)
- r/LocalLLaMA subreddit — there's genuine local-deployment interest in
  lattice-based KV compression right now
