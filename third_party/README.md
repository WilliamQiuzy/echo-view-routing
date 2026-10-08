# third_party

Vendored external code. Policy: clone-only at a pinned commit, never edited; adapters live in `echo_routing/`.

| Project | Source | Pinned commit | Where |
|---|---|---|---|
| STFM (Spatio-Temporal Fusion Model, EV9V authors) | https://github.com/bgx666/stfm | `532f60b2733a3442dcccb59099e00eba07bf293d` | server: `third_party/stfm/` (gitignored locally) |
| FACT (Lu and Elhamifar, CVPR 2024) | https://github.com/ZijiaLewisLu/CVPR2024-FACT | `7bd81bda2b84618a3e23876a2637a82f24881122` | server: `third_party/fact/` (gitignored locally; a local clone of the same commit serves the CPU tests); needs yacs 0.1.8; MIT licence |
| EchoSlicer (Vukadinovic et al., arXiv 2511.15946) | https://github.com/echonet/3d-echo | `78f7085104492d5229b1bb96b7748ad9d950e53d` | server: `third_party/3d-echo/` (pristine clone) and `third_party/3d-echo_run/` (copy with the frozen EchoPrime view-classifier weights, the EchoNet-Dynamic LV weights and its own `uv` environment); the repository has no LICENSE file |

Re-clone on the server with `git clone https://github.com/bgx666/stfm.git third_party/stfm && git -C third_party/stfm checkout 532f60b`.

## Patches

`patches/` holds the only changes ever made to vendored code; each is applied to a fresh copy, never to the pinned clone.

- `ms-tcn-py3.patch`, `asformer-py3.patch`: Python 3 compatibility only.
- `stfm-numeric-frame-order.patch` (2026-09-30): the official STFM loader lists frames with `sorted(os.listdir(...))`.
  EV9V names frames `0.jpg, 1.jpg, …` without zero padding, so that order shuffles frames within each clip. The patch
  sorts by the number in the name. The pinned clone stays unpatched so that the reproduction of the authors' numbers
  remains exact. A fair temporal baseline uses a patched copy:
  `cp -r third_party/stfm third_party/stfm_numeric && git -C third_party/stfm_numeric apply ../patches/stfm-numeric-frame-order.patch`.
