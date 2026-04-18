# AgentMemorySystem

This repository currently carries multiple AMS snapshots. The `v331` branch is the branch for the `v3.31` HuggingFace-backbone version.

## v331 Branch

On `v331`, the main implementation lives in:

- `AgentMemorySystem.py`

The black-box audit materials for `v3.31` live in:

- `V331_BLACKBOX_TEST_SPEC.md`
- `v331_blackbox_eval.py`

## Audit Policy

The `v3.31` branch follows the same audit policy described in `V331_BLACKBOX_TEST_SPEC.md`:

- no `mock`
- no `fallback`
- no `overfit`
- no simplified replacement path
- external black-box runner only

## How To Run

Install runtime dependencies first:

```bash
pip install torch transformers
```

Then run the external audit:

```bash
python v331_blackbox_eval.py
```

Generated reports are written under:

```text
reports/v331_blackbox/
```

## Notes

- The audit runner treats `AgentMemorySystem.py` as the system under test.
- The runner does not call the module-internal `test()` entrypoint.
- The specification document records the exact test conditions, case list, prompts, corpora, and pass criteria used for the `v3.31` audit.
