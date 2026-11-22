"""Optional-dependency management for su.

The su package keeps a *lightweight core* (audio format conversion, waveform
handling, cover-art embedding, image handling) installable out of the box, and
places heavier or more fragile capabilities behind extras:

- ``su[speech]`` — speech recognition + text-to-speech
  (``SpeechRecognition``, ``pyttsx3``, ``pyaudio``).
- ``su[ml]`` — audio analysis / ML feature extraction
  (``librosa``, ``soundfile``).

Modules that use an optional dependency import it *lazily* through
:func:`require` so that ``import su`` never hard-fails and each feature reports a
precise, actionable error only when it is actually invoked.

Example:
    >>> np = require('numpy', extra='ml', feature='waveform analysis')
    >>> int(np.array([1, 2, 3]).sum())
    6
"""

import importlib
from types import ModuleType

# Map an importable module name -> (extra that provides it, PyPI dist name).
# Used to render an informative install hint when the dependency is missing.
_MODULE_TO_EXTRA: dict[str, tuple[str, str]] = {
    "speech_recognition": ("speech", "SpeechRecognition"),
    "pyttsx3": ("speech", "pyttsx3"),
    "pyaudio": ("speech", "pyaudio"),
    "librosa": ("ml", "librosa"),
    "soundfile": ("ml", "soundfile"),
    "faster_whisper": ("transcribe", "faster-whisper"),
    "faster_whisper.vad": ("transcribe", "faster-whisper"),
    "demucs.pretrained": ("transcribe", "demucs"),
    "demucs.apply": ("transcribe", "demucs"),
    "torch": ("transcribe", "torch"),
    "torchaudio": ("transcribe", "torchaudio"),
    "torchaudio.functional": ("transcribe", "torchaudio"),
}


class MissingDependencyError(ImportError):
    """Raised when an optional dependency needed for a feature is not installed.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working.
    """


def require(
    module_name: str,
    *,
    extra: str | None = None,
    feature: str | None = None,
) -> ModuleType:
    """Import and return ``module_name``, or raise an informative error.

    Args:
        module_name: The importable module name (e.g. ``'speech_recognition'``).
        extra: The su extra that provides this dependency (e.g. ``'speech'``).
            If ``None``, it is looked up from a built-in table when possible.
        feature: Human-readable name of the feature needing the dependency,
            used to make the error message actionable.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module cannot be imported, with a message
            telling the user exactly what to ``pip install``.

    Example:
        >>> mod = require('io')  # a stdlib module always resolves
        >>> mod.__name__
        'io'
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        known_extra, dist_name = _MODULE_TO_EXTRA.get(module_name, (extra, module_name))
        resolved_extra = extra or known_extra
        feature_clause = f" needed for {feature}" if feature else ""
        if resolved_extra:
            hint = f'pip install "su[{resolved_extra}]"'
        else:
            hint = f"pip install {dist_name}"
        raise MissingDependencyError(
            f"Missing optional dependency '{module_name}'{feature_clause}. "
            f"Install it with:  {hint}"
        ) from e
