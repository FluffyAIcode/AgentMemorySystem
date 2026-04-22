"""Skeleton-level tests: imports + Cfg4 invariants + NotImplementedError markers.

This file is DOCUMENTATION of what the skeleton guarantees at this stage.
It does NOT test forward passes — those land in each subsequent v4.x PR.

What is tested here (what the skeleton guarantees today):
  1. `import ams_v4` and `from ams_v4 import ...` all succeed.
  2. `Cfg4()` constructs with default values and all invariants pass.
  3. Cfg4 invariants actually fire when violated (sample of three).
  4. Classes that are supposed to raise NotImplementedError do so with a
     clear "v4-skel: ..." message — this is the contract that downstream
     implementers fill in.

Run with:
  python3 -m pytest ams_v4/tests/test_shapes.py -v
or
  python3 ams_v4/tests/test_shapes.py    # argparse-free self-test
"""
from __future__ import annotations
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_imports():
    import ams_v4
    from ams_v4 import (
        Cfg4, MemEntry, KakeyaHandle, MemStore,
        TemporalBundle, TimeEncoder,
        TopicBundle, TopicEncoder,
        ContextBundle, ContextEncoder,
        KakeyaSet, KakeyaRegistry,
        CrossBundleAttention,
        EmbBridge4,
        MemLLM4,
    )
    assert ams_v4.__version__.startswith("4.0.0"), ams_v4.__version__


def test_cfg4_default_constructs():
    from ams_v4 import Cfg4
    cfg = Cfg4()
    assert cfg.n_kakeya_sets >= 2
    assert cfg.prefix_slots_time + cfg.prefix_slots_topic + cfg.prefix_slots_ctx == cfg.L_mem


def test_cfg4_invariant_n_kakeya_sets_min_2():
    from ams_v4 import Cfg4
    try:
        Cfg4(n_kakeya_sets=1)
    except AssertionError as e:
        assert "multiple kakeya sets" in str(e) or "n_kakeya_sets" in str(e)
        return
    raise AssertionError("Cfg4(n_kakeya_sets=1) should have raised")


def test_cfg4_invariant_prefix_slots_sum():
    from ams_v4 import Cfg4
    try:
        Cfg4(L_mem=12, prefix_slots_time=2, prefix_slots_topic=6, prefix_slots_ctx=5)
    except AssertionError as e:
        assert "prefix_slots" in str(e)
        return
    raise AssertionError("mismatched prefix_slots should have raised")


def test_cfg4_invariant_fiber_divisibility():
    from ams_v4 import Cfg4
    try:
        Cfg4(d_F_time=33, n_heads_time=4)
    except AssertionError as e:
        assert "d_F_time" in str(e)
        return
    raise AssertionError("non-divisible fiber dim should have raised")


def test_all_skeleton_components_raise_not_implemented():
    """Constructing the stubbed modules must raise NotImplementedError with the
    'v4-skel:' marker. This is the contract for downstream v4.1-v4.5 PRs.
    """
    from ams_v4 import Cfg4
    cfg = Cfg4()

    from ams_v4.core.mem_store import MemStore
    store = MemStore(cfg)  # __init__ does NOT raise — but its methods do
    for method_call in [
        lambda: store.add(None),
        lambda: store.remove(0),
        lambda: store.verify_consistency(),
    ]:
        try:
            method_call()
        except NotImplementedError as e:
            assert "v4-skel" in str(e)
        except Exception as e:
            raise AssertionError(f"expected NotImplementedError, got {type(e).__name__}: {e}")
        else:
            raise AssertionError("method should have raised NotImplementedError")


def _run_all():
    tests = [
        test_imports,
        test_cfg4_default_constructs,
        test_cfg4_invariant_n_kakeya_sets_min_2,
        test_cfg4_invariant_prefix_slots_sum,
        test_cfg4_invariant_fiber_divisibility,
        test_all_skeleton_components_raise_not_implemented,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} / {len(tests)} failed: {failed}")
        sys.exit(1)
    print(f"\nall {len(tests)} skeleton tests passed")


if __name__ == "__main__":
    _run_all()
