#!/usr/bin/env python3
"""
AMS v3.31 — Semantic Audit Entry
================================

This suite provides a repo-local semantic entrypoint aligned to the
`v3.31` public interface. It reuses the same external black-box audit
functions used by `v331_blackbox_eval.py`, but focuses on semantic and
generation-quality cases only.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import v331_blackbox_eval as audit


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


SEMANTIC_CASES: List[Tuple[str, Callable[[int], Dict], int]] = [
    ("counterfactual_memory_influence", audit.counterfactual_memory_influence, 31),
    ("semantic_memory_grounding", audit.semantic_memory_grounding, 33),
    ("semantic_memory_counterfactual_pairs", audit.semantic_memory_counterfactual_pairs, 35),
    ("degeneration_quality", audit.degeneration_quality, 36),
    ("prompt_diversity_without_memory", audit.prompt_diversity_without_memory, 37),
    ("prefix_logit_drift_audit", audit.prefix_logit_drift_audit, 38),
    ("retrieval_topk_semantic_shift", audit.retrieval_topk_semantic_shift, 39),
    ("repetition_segment_audit", audit.repetition_segment_audit, 40),
    ("prefix_stepwise_drift_trajectory", audit.prefix_stepwise_drift_trajectory, 44),
    ("retrieval_generation_alignment_audit", audit.retrieval_generation_alignment_audit, 45),
    ("retrieval_prefix_decode_correlation_audit", audit.retrieval_prefix_decode_correlation_audit, 46),
    ("cheating_heuristics", audit.cheating_heuristics, 47),
    ("stepwise_label_mass_alignment_audit", audit.stepwise_label_mass_alignment_audit, 48),
]


def run_case(name: str, fn: Callable[[int], Dict], seed: int) -> Dict:
    print(f"[case:start] {name}", flush=True)
    try:
        result = fn(seed)
        if "passed" not in result:
            result["passed"] = True
        result["error"] = None
        print(f"[case:done] {name} passed={result['passed']}", flush=True)
        return result
    except Exception as exc:  # pragma: no cover - reporting path
        print(f"[case:done] {name} passed=False error={type(exc).__name__}: {exc}", flush=True)
        return {
            "passed": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _detail(payload: Dict) -> str:
    if payload.get("error"):
        return payload["error"]["message"]
    trimmed = {k: v for k, v in payload.items() if k not in {"passed", "error"}}
    return json.dumps(trimmed, ensure_ascii=False)[:500]


def main() -> int:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  AMS v3.31 — Semantic Audit Entry")
    print(f"{sep}")
    print("\n[Running semantic subset of the external v3.31 audit]")
    t0 = time.time()

    results = {name: run_case(name, fn, seed) for name, fn, seed in SEMANTIC_CASES}
    checks = [CheckResult(name=name, passed=payload["passed"], detail=_detail(payload))
              for name, payload in results.items()]

    print("\n## Summary")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"- {status} {check.name}: {check.detail}")

    elapsed = time.time() - t0
    passed = sum(c.passed for c in checks)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Passed: {passed}/{len(checks)}")
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
