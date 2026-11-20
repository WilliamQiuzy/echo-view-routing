"""The reference system, as fixed on 2026-11-13: what it is called, where its weights are and how it decodes.

The system is three training seeds of one window model ("C2"), run as one probability ensemble. Nothing in this file is
learned at run time. The decoder setting was chosen once on the EV9V calibration half and is part of the system.
"""
from __future__ import annotations

NAME = "EchoRouter"  # the system's name
VERSION = "C2_ens_v1"  # the identifier of its weights and decoder setting, as the project's records use it
SEEDS = (0, 1, 2)

# The change-point decoder ("B3") of the ensemble: half window in 10 Hz samples, threshold on the Jensen-Shannon
# score (the 0.999 quantile of the scores inside the native recordings of the EV9V calibration half), and the radius
# inside which a weaker peak is dropped.
B3 = {"half_window": 7, "threshold": 0.1737918038235492, "nms_radius": 5}

# The rule of the uncertain output (`echoview segment --uncertain`): the fields of echoview.uncertain.Rule. The band is
# in 10 Hz samples. The rule was chosen on the EV9V calibration half and checked on the selection half on 2026-11-20
# (docs/UNCERTAIN.md, sections 7 and 8).
UNCERTAIN = {"band": 3, "f_low": 2.0, "f_high": 1.25, "margin": 0.1, "seeds": False}

# Where the weights are published. MODEL_CARD.md says under which licence.
HUB_REPO = "WilliamQiu123/echo-view-routing"
HUB_REVISION = "main"

# SHA-256 of the start weights of training stage 3: the stage-2 model of every seed (`training/stage2_s<seed>.pt`).
START_SHA256 = {
    0: "9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc",
    1: "c8b09ecaf286c7b66fc0eec69fdd102f98930aaa657a7604e3204c3845a8e172",
    2: "f95fed7ba78cc0b5e2474c71a156356507b4588a4b324957fd5b53fe66ee8472",
}

# SHA-256 of every file that may serve as a seed's weights. `final.pt` is the frozen training output; `model.safetensors`
# holds the same tensors in the safetensors format.
WEIGHT_SHA256 = {
    0: {
        "final.pt": "82b85f072326fc475aea6253fd5995221e12d47eead0e1d399835b92feb370d7",
        "model.safetensors": "3319458a3d4a63c3eb442bb11ccf47a9ba60173e882f3bba2bbc3074f6a9b612",
    },
    1: {
        "final.pt": "31747977a3d3042f94e506e44f76423750b9275183611bc5f804eb6bc90fd6ff",
        "model.safetensors": "498f72bc478ca69c1ee2ca67c8bda754f2bc5f0379beccece3834448d7e14e49",
    },
    2: {
        "final.pt": "3c27df1e616ba3791211aec19ecdc00b2bb225e6c2e0d8a421eca918e4e07194",
        "model.safetensors": "ceab7f07bbd53e1ce96241442c07cb8e45e86885c612c829d0af5c324726cc23",
    },
}
