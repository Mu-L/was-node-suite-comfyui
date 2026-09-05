# EMA-VFI, vendored

Frame interpolation network from **Extracting Motion and Appearance via Inter-Frame Attention for
Efficient Video Frame Interpolation** (CVPR 2023), Zhang, Zhu, Wang, Chen, Wu, Wang.

- Upstream: <https://github.com/MCG-NJU/EMA-VFI>
- Taken at commit `75b6f6a889e695df875e103374040d47a4cfac7c` (2023-05-29)
- Licence: Apache-2.0, retained verbatim in `LICENSE` beside this file

Vendored rather than reimplemented because the released checkpoints are bare `state_dict`
pickles: every parameter name has to match the original module tree exactly or the weights will
not load. Rewriting the network would have meant guessing at those names.

**No weights are bundled.** They are downloaded by the user and placed in
`ComfyUI/models/EMA-VFI`. See `docs/MODELS.md`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `feature_extractor.py` | `model/feature_extractor.py` | import line only |
| `flow_estimation.py` | `model/flow_estimation.py` | import line, two device calls |
| `refine.py` | `model/refine.py` | import line, one dead line removed |
| `warplayer.py` | `model/warplayer.py` | rewritten, same arithmetic |

## Every change made

1. **`from timm.models.layers import ...` → `from ._torch_compat import ...`**, in
   `feature_extractor.py` and `refine.py`. timm is not installed in the environments this pack
   targets. `_torch_compat.py` supplies the three names from torch alone: `trunc_normal_` is
   bound straight to `torch.nn.init.trunc_normal_`, which takes the same arguments in the same
   order; `to_2tuple` and `DropPath` are reimplemented. None of the three holds a parameter, and
   `DropPath` is `nn.Identity` in the released configuration anyway, so no checkpoint key moves.

2. **`torch.full(...).cuda()` → `torch.full(..., device=mf[-1-i].device)`**, twice in
   `flow_estimation.py` (`calculate_flow` and `forward`). Upstream hardcoded a GPU, so the
   network could not run at all on a CPU-only ComfyUI. Identical behaviour on CUDA.

3. **`warplayer.py` rewritten.** Upstream picked its device once at import from
   `torch.cuda.is_available()` and cached sampling grids in a module-level dict keyed by shape.
   The first meant a CPU-only install was handed a CUDA grid; the second grows without limit as
   a workflow feeds it new resolutions. The grid is now built per call on the flow's own device.
   The arithmetic is unchanged, including sizing the grid from the flow while normalising against
   the input.

4. **One dead line dropped** from `refine.py`: a module-level `device = torch.device(...)` that
   nothing in the file read.

Nothing else was touched. No class, function, attribute or parameter name differs from upstream,
which is what keeps the released `state_dict` loadable with `strict=True`.

`modules/vendor/` holds code this repository did not write. It is kept as close to upstream as
loading the released weights allows, and is not restyled.
