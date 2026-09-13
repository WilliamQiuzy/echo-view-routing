# third_party

Vendored external code. Policy: clone-only at a pinned commit, never edited; adapters live in `echo_routing/`.

| Project | Source | Pinned commit | Where |
|---|---|---|---|
| STFM (Spatio-Temporal Fusion Model, EV9V authors) | https://github.com/bgx666/stfm | `532f60b2733a3442dcccb59099e00eba07bf293d` | server: `third_party/stfm/` (gitignored locally) |

Re-clone on the server with `git clone https://github.com/bgx666/stfm.git third_party/stfm && git -C third_party/stfm checkout 532f60b`.
