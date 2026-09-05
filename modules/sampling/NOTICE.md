# SPEED

**Spectral Progressive Diffusion**, implemented here from the paper's equations. No upstream
code is carried in this directory.

- Paper and reference implementation: <https://github.com/howardhx/speed>
- Licence there: MIT, Copyright (c) 2026 Howard Xiao
- Read at commit `ca7801c9bdffe681742e9592345bcf4885959be5`

The equations are the paper's: state rescaling, timestep alignment, activation time and the
delta-optimal transition times. The spectral expansions run in torch on the device the latent
is already on, and add no dependency.

## Where this differs from the reference implementation

| | Here |
|---|---|
| Noise draw | A seeded `torch.Generator` on the CPU, which is how ComfyUI draws its sampling noise. The same seed gives different numbers from upstream's |
| Precision | Single precision or the input's own, whichever is wider |
| Amplitude and exponent | Taken as values, with **Latent Power Spectrum** to measure them from a model's own latents. The two published pairs are starting points rather than a menu |
| A latent that is not a grid | Refused by name. Growth is over the trailing two axes, and a flat token sequence has nothing to grow |

## Measuring the pair for a model

`A` and `delta` only mean anything together. Measure both from the same place with **Latent
Power Spectrum** and tune `delta` against what was measured; a measured `A` beside a `delta`
chosen for a published one transitions at the wrong time.

The exponent carries across content and the amplitude does not. For Wan 2.1 T2V-1.3B, whose
published pair is `(219.485, 2.4227)`, six ordinary images through that model's own VAE fit
`beta` between 2.43 and 2.59 and `A` between 454 and 2065.
