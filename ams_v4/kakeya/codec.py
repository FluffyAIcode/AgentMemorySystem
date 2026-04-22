"""KakeyaCodecV4 — unified codec interface (thin wrapper over KakeyaRegistry).

v3.46 had `KakeyaCodec` + `KakeyaMemLLM` wrapper; v4 merges the concerns:
MemLLM4 holds a KakeyaRegistry directly, and any external caller that needs
a "codec-like" API uses this thin facade.

Kept as a separate module purely for migration continuity — downstream tools
or tests that expected `from kakeya_codec import KakeyaCodec` can switch to
`from ams_v4.kakeya.codec import KakeyaCodecV4` with minimal edits.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import KakeyaHandle
from ams_v4.core.types import Tensor
from ams_v4.kakeya.registry import KakeyaRegistry


class KakeyaCodecV4:
    """Facade over a KakeyaRegistry. Use this when integrating a KakeyaRegistry
    into a module that expects a codec-shaped object.
    """

    def __init__(self, cfg: Cfg4, registry: Optional[KakeyaRegistry] = None):
        self.cfg = cfg
        self.registry = registry if registry is not None else KakeyaRegistry(cfg)

    def encode(self, fields: Dict[str, Tensor]) -> KakeyaHandle:
        return self.registry.encode_memory_fields(fields)

    def decode(self, handle: KakeyaHandle, field_name: str,
               device: Optional[torch.device] = None) -> Optional[Tensor]:
        return self.registry.decode_field(handle, field_name, device=device)

    @property
    def n_sets(self) -> int:
        return len(self.registry.sets)
