#!/usr/bin/env python
"""Print device, VRAM, and the memory cap we will apply (co-tenant safety check)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction  # noqa: E402

if not torch.cuda.is_available():
    print("cuda: unavailable"); sys.exit(0)
free, total = torch.cuda.mem_get_info(0)
frac = gpu_memory_fraction()
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB cap={frac} -> {frac*total/2**30:.1f}GiB")
