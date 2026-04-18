#!/usr/bin/env python3
"""
AMS v3.31 — Internal Black-Box Entry
====================================

This entrypoint exposes the branch-local internal `v3.31` test suite that
ships inside `AgentMemorySystem.py`.

Rules:
  - no mocks
  - no fallback logic
  - no source modifications during the run
  - real HuggingFace causal LM path only
"""

from __future__ import annotations

import sys
import time

import AgentMemorySystem as ams


def main() -> int:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  AMS v3.31 — Internal Black-Box Entry")
    print(f"{sep}")
    print("\n[Running AgentMemorySystem.test()]")
    t0 = time.time()
    ok = ams.test()
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
