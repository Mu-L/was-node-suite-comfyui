"""Measuring how a model's latents distribute power across frequency."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import speed


class LatentPowerSpectrum(io.ComfyNode):
    """Fit a power law to a latent's radial spectrum."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLatentPowerSpectrum",
            display_name="Latent Power Spectrum",
            search_aliases=[
                'WASLatentPowerSpectrum',
                "Latent Power Spectrum",
                "measure latent spectrum",
                "power law fit",
                "spectrum amplitude beta",
                "SPEED spectrum",
            ],
            category="WAS Suite/Latent",
            description=(
                "Measure the amplitude and falloff of a latent's radial power spectrum, the "
                "two values SPEED Sampler schedules its resolution changes from. Encode "
                "ordinary content through the model's own VAE and wire the outputs across. "
                "Average over several images: the falloff settles quickly, the amplitude "
                "moves with the content and wants more of it."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip=(
                        "A latent from the model being sampled, encoded from ordinary content "
                        "rather than noise. A batch or a whole video measures more steadily "
                        "than a single frame."
                    ),
                ),
                io.Float.Input(
                    "low",
                    default=0.05,
                    min=0.0,
                    max=0.9,
                    step=0.01,
                    tooltip=(
                        "Where to start fitting, as a fraction of the highest frequency the "
                        "latent can hold. The lowest frequencies rest on a handful of "
                        "coefficients and do not follow the power law, so they are skipped."
                    ),
                ),
                io.Float.Input(
                    "high",
                    default=0.5,
                    min=0.1,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Where to stop fitting, as a fraction of the highest frequency. Near "
                        "that limit the spectrum rolls off for reasons to do with the encoder "
                        "rather than the content."
                    ),
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="amplitude",
                    tooltip=(
                        "The A of P(w) = A * w ** -beta. Wire to SPEED Sampler, and tune its "
                        "delta against this rather than against a published figure: this runs "
                        "larger, and it moves with the content measured."
                    ),
                ),
                io.Float.Output(
                    display_name="beta",
                    tooltip=(
                        "How fast the spectrum falls away. Steady across content and directly "
                        "comparable to a published figure, unlike the amplitude."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip="The fitted pair as text, for a note or a filename.",
                ),
            ],
        )

    @classmethod
    def execute(cls, samples, low: float, high: float) -> io.NodeOutput:
        """Fit the spectrum and report the pair.

        Raises:
            ValueError: The band is empty or inverted, the latent has no spatial grid, or it is
                too small to fit a line through.
        """
        if low >= high:
            raise ValueError(
                f"low ({low:g}) has to be below high ({high:g}); they are the ends of the band "
                f"the power law is fitted over."
            )

        amplitude, beta = speed.fit_power_spectrum(samples["samples"], low=low, high=high)
        summary = f"A={amplitude:.6f} beta={beta:.6f}"
        return io.NodeOutput(amplitude, beta, summary)
