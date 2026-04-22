"""AMS v4 — realigned architecture.

Abstract spec (the invariant this package must honor):

    Multiple Kakeya sets compress the full context data. These Kakeya sets
    are linked on different fiber bundles. The fiber bundles carry memory
    encoding around time, topic, and background (context). An attention
    mechanism forms the current context window.

Public surface kept small on purpose — most users only need Cfg4 + MemLLM4.
See ARCHITECTURE_v4.md for the abstract-to-concrete mapping and invariants.
"""
from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import MemEntry, KakeyaHandle
from ams_v4.core.mem_store import MemStore
from ams_v4.bundles.temporal import TemporalBundle, TimeEncoder
from ams_v4.bundles.topic import TopicBundle, TopicEncoder
from ams_v4.bundles.context import ContextBundle, ContextEncoder
from ams_v4.kakeya.set import KakeyaSet
from ams_v4.kakeya.registry import KakeyaRegistry
from ams_v4.attention.cross_bundle import CrossBundleAttention
from ams_v4.projection.bridge import EmbBridge4
from ams_v4.bridge.memllm import MemLLM4

__all__ = [
    "Cfg4",
    "MemEntry",
    "KakeyaHandle",
    "MemStore",
    "TemporalBundle",
    "TimeEncoder",
    "TopicBundle",
    "TopicEncoder",
    "ContextBundle",
    "ContextEncoder",
    "KakeyaSet",
    "KakeyaRegistry",
    "CrossBundleAttention",
    "EmbBridge4",
    "MemLLM4",
]

__version__ = "4.0.0.dev0-skeleton"
