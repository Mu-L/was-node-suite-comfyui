"""What pip would do with a requirements file, and doing it.

Stdlib only and importing nothing from the pack, so ``prestartup_script.py`` can load it by
path before any custom node is imported.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "DISTRIBUTIONS",
    "RESOLVE_TIMEOUT",
    "Change",
    "Resolution",
    "apply_requirements",
    "environment_kind",
    "install_instruction",
    "missing_from",
    "pip_command",
    "requirement_names",
    "resolve_requirements",
]

#: Seconds a resolve may take before it is abandoned. A cold index over a slow link is the
#: slow case; the answer is only ever used to decide whether to install.
RESOLVE_TIMEOUT = 300

#: Seconds an install may take. Wheels for a document toolchain run to tens of megabytes.
INSTALL_TIMEOUT = 1800

#: The end of a requirement line: an extra, a version, a marker or a comment.
_BOUNDARY = re.compile(r"[\[<>=!~;#\s]")

#: Module a node imports -> the distribution that provides it, where the two are spelled
#: differently.
DISTRIBUTIONS = {
    "docx": "python-docx",
    "git": "GitPython",
    "huggingface_hub": "huggingface-hub",
}

#: The same table read the other way, for a requirements line that names a distribution and
#: has to be answered with the module a node would import.
_MODULES = {distribution: module for module, distribution in DISTRIBUTIONS.items()}


class Change(NamedTuple):
    """One installed distribution that would be replaced.

    Attributes:
        name: Distribution name as pip spells it.
        have: Version installed now.
        want: Version that would take its place.
    """

    name: str
    have: str
    want: str

    @property
    def direction(self) -> str:
        """``upgrade``, ``downgrade``, or ``change`` where the two cannot be ordered."""
        first, second = _ordered(self.have), _ordered(self.want)
        if first is None or second is None or first == second:
            return "change"
        return "upgrade" if second > first else "downgrade"

    def __str__(self) -> str:
        return f"{self.direction} {self.name} {self.have} to {self.want}"


def _ordered(version: str):
    """One version as something comparable, or None where it cannot be read.

    Args:
        version: A version string as pip reports it.

    Returns:
        A tuple of the leading numeric release parts, or None.
    """
    parts = []
    for chunk in re.split(r"[._-]", version.strip()):
        digits = re.match(r"\d+", chunk)
        if digits is None:
            break
        parts.append(int(digits.group()))
    return tuple(parts) or None


def pip_command(*arguments: str) -> str:
    """A pip install command line for the interpreter running this, ready to paste.

    Args:
        arguments: What to install: one requirement, or ``-r`` and a file path.

    Returns:
        The command, quoted wherever an argument holds a space.
    """
    # The python running ComfyUI, not a bare pip: on a portable install the pip on PATH
    # belongs to another interpreter, and what it installs lands where ComfyUI never looks.
    # -s under no_user_site is how ComfyUI names this command for its own requirements.
    if not sys.executable:
        prefix = ["pip"]
    elif sys.flags.no_user_site:
        prefix = [sys.executable, "-s", "-m", "pip"]
    else:
        prefix = [sys.executable, "-m", "pip"]
    return " ".join(
        f'"{part}"' if " " in part else part
        for part in (*prefix, "install", *arguments)
    )


class Resolution(NamedTuple):
    """What pip answers for one requirements file.

    Attributes:
        additions: ``name==version`` for each distribution that is not installed.
        changes: One :class:`Change` per installed distribution that would be replaced.
        failure: Why pip could not answer, or an empty string.
    """

    additions: tuple[str, ...]
    changes: tuple[Change, ...]
    failure: str

    @property
    def safe(self) -> bool:
        """Whether only additions are needed, so nothing already installed moves."""
        return not self.failure and not self.changes

    @property
    def wanted(self) -> bool:
        """Whether anything would be installed at all."""
        return bool(self.additions or self.changes)


def environment_kind() -> str:
    """Which kind of python install is running, named the way its user would name it.

    Returns:
        ``portable``, ``desktop``, ``conda``, ``venv`` or ``system``.
    """
    import os

    executable = (sys.executable or "").replace("\\", "/").lower()
    if "python_embeded" in executable or "python_embedded" in executable:
        return "portable"
    if os.environ.get("CONDA_PREFIX"):
        return "conda"
    if "/comfyui/.venv/" in executable or "comfyui-electron" in executable:
        return "desktop"
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return "venv"
    return "system"


def install_instruction(path: Path) -> str:
    """How to install a requirements file by hand, for the install that is running.

    Args:
        path: The requirements file.

    Returns:
        One sentence naming the kind of install and the command to run in it. The command
        names the interpreter by full path, so it works from any directory and needs no
        environment activated first.
    """
    command = pip_command("-r", str(path))
    where = {
        "portable": "this is a portable install, so use its bundled python",
        "desktop": "this is the ComfyUI desktop app, so use the python inside it",
        "conda": "this is a conda environment, so use its python rather than a bare pip",
        "venv": "this is a virtual environment, so use its python rather than a bare pip",
        "system": "use the python that runs ComfyUI",
    }[environment_kind()]
    return f"{where}: {command}"


def requirement_names(path: Path) -> tuple[str, ...]:
    """The distribution names one requirements file asks for.

    Args:
        path: A requirements file. A ``-r`` line is not followed.

    Returns:
        Names in file order, without extras, versions, markers or comments.
    """
    found = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines:
        text = line.strip()
        if not text or text.startswith(("#", "-")):
            continue
        name = _BOUNDARY.split(text, 1)[0].strip()
        if name:
            found.append(name)
    return tuple(found)


def module_for(name: str) -> str:
    """The module a node imports for one requirement name.

    Args:
        name: A distribution name as a requirements file spells it.

    Returns:
        The module, from :data:`DISTRIBUTIONS` where the two differ and the name itself
        otherwise, with dashes read as underscores.
    """
    return _MODULES.get(name) or name.replace("-", "_")


def missing_from(names) -> tuple[str, ...]:
    """Which of these requirements nothing installed provides.

    Any distribution providing the module satisfies it.

    Args:
        names: Distribution names, as :func:`requirement_names` answers them.

    Returns:
        The subset nothing provides.
    """
    from importlib import metadata

    # onnxruntime comes from the CPU or the GPU build, and a second one writes over the
    # first.
    try:
        provided = metadata.packages_distributions()
    except Exception:
        provided = {}

    absent = []
    for name in names:
        if provided.get(module_for(name)):
            continue
        try:
            metadata.version(name)
        except Exception:
            absent.append(name)
    return tuple(absent)


def resolve_requirements(
    path: Path, python: str | None = None, timeout: int = RESOLVE_TIMEOUT
) -> Resolution:
    """What pip would do with this file, without doing it.

    Args:
        path: The requirements file.
        python: Interpreter whose environment is resolved against. The running one by default.
        timeout: Seconds the resolve may take.

    Returns:
        A :class:`Resolution`. A resolve that fails carries its reason in ``failure`` and
        nothing in the other two fields, so a caller that installs only a safe answer
        does nothing.
    """
    from importlib import metadata

    command = [
        python or sys.executable, "-m", "pip", "install",
        "--dry-run", "--no-input", "--disable-pip-version-check", "--quiet",
        "--report", "-", "-r", str(path),
    ]
    try:
        finished = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as error:
        return Resolution((), (), f"pip could not be run ({type(error).__name__}: {error})")
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout or "").strip().splitlines()
        return Resolution(
            (), (), f"pip refused the file: {detail[-1] if detail else 'no reason given'}"
        )

    try:
        report = json.loads(finished.stdout or "")
    except ValueError as error:
        return Resolution((), (), f"pip's report could not be read ({error})")
    if not isinstance(report, dict):
        return Resolution((), (), "pip answered something that is not a report")

    additions, changes = [], []
    for entry in report.get("install", []):
        info = entry.get("metadata") or {}
        name, want = info.get("name"), info.get("version")
        if not name or not want:
            continue
        try:
            have = metadata.version(name)
        except Exception:
            additions.append(f"{name}=={want}")
            continue
        if have != want:
            changes.append(Change(name, have, want))
    return Resolution(tuple(additions), tuple(changes), "")


def apply_requirements(
    path: Path, python: str | None = None, timeout: int = INSTALL_TIMEOUT
) -> str:
    """Install a requirements file.

    Args:
        path: The requirements file.
        python: Interpreter to install into. The running one by default.
        timeout: Seconds the install may take.

    Returns:
        An empty string where it worked, otherwise why it did not.
    """
    command = [
        python or sys.executable, "-m", "pip", "install",
        "--no-input", "--disable-pip-version-check", "--no-warn-script-location",
        "-r", str(path),
    ]
    try:
        finished = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"pip could not be run ({type(error).__name__}: {error})"
    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout or "").strip().splitlines()
        return detail[-1] if detail else f"pip exited {finished.returncode}"
    return ""
