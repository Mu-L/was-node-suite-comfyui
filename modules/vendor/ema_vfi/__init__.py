"""EMA-VFI's network, vendored from https://github.com/MCG-NJU/EMA-VFI under Apache-2.0.

See ``NOTICE.md`` beside this file for the commit it was taken at and every change made to it.

Nothing is imported here on purpose. ``feature_extractor`` and ``flow_estimation`` both pull in
torch and build large module trees, and ComfyUI imports every pack at startup, so the wrapper in
``modules/model/frame_interpolation.py`` imports them from inside a function instead. A workflow
that never interpolates a frame never pays for them.
"""
