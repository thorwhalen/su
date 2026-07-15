"""Flexible input/output casting for audio (and text).

Postel's principle applied to audio: *be liberal in what you accept, strict in
what the base functions require.* This module provides a small
:mod:`i2.castgraph`-based transformation layer that normalises the many
*representations* an audio resource can take — file paths, encoded bytes,
file-like objects, numpy waveforms, plain sample lists, or a
:class:`pydub.AudioSegment` — into the strict types the base functions consume,
and lets callers control the *output* format when it matters (and pipe raw bytes
when it does not).

Canonical hub
-------------
:class:`pydub.AudioSegment` is the canonical in-memory audio type (the "hub"):
it carries sample rate / channels / sample width, constructs from raw PCM or any
ffmpeg-decodable file/bytes, and exports to any format. Every *input* kind
converts **to** the hub; every *output* is derived **from** it. castgraph composes
multi-hop routes automatically (e.g. ``audio_path -> AudioSegment -> wav_source``).

Kinds
-----
Input kinds:  ``'audio_path'``, ``'audio_bytes'``, ``'audio_file_like'``,
``'waveform'``, ``'waveform_sr'``, ``'sample_list'``, and the hub
:class:`~pydub.AudioSegment`.

Output kinds: ``'wav_source'``, ``'wav_bytes'``, ``'waveform_sr'`` (+ the hub).

Public API
----------
- :func:`to_audio_segment` — universal ingress to the hub.
- :func:`to_wav_source` / :func:`to_wav_bytes` — WAV for :mod:`speech_recognition`.
- :func:`to_waveform` — ``(np.ndarray, sample_rate)`` for DSP/ML.
- :func:`export_audio` — the egress: choose the output format, or get bytes.
- :func:`to_text` — flexible text ingress.
- :data:`audio_graph` — the underlying :class:`~i2.castgraph.TransformationGraph`.

Note:
    Decoding/encoding non-WAV formats (mp3, flac, m4a, ...) uses **ffmpeg** via
    pydub. Install it separately (e.g. ``brew install ffmpeg`` /
    ``apt-get install ffmpeg``). WAV and raw-sample conversions need no ffmpeg.

Example:
    >>> import numpy as np
    >>> from su.casting import to_audio_segment, to_wav_bytes, to_waveform
    >>> # A numpy waveform (int16 PCM) becomes an AudioSegment...
    >>> samples = (np.sin(np.linspace(0, 6.28, 8000)) * 16000).astype(np.int16)
    >>> seg = to_audio_segment(samples, sample_rate=8000)
    >>> seg.frame_rate, seg.channels
    (8000, 1)
    >>> # ...exports to WAV bytes (no ffmpeg needed)...
    >>> to_wav_bytes(seg)[:4]
    b'RIFF'
    >>> # ...and round-trips back to a (waveform, sample_rate) pair.
    >>> wf, sr = to_waveform(seg, dtype='int16')
    >>> wf.shape, sr
    ((8000,), 8000)
"""

import io
import os
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from i2.castgraph import TransformationGraph, ConversionError

# Default sample rate assumed for raw sample inputs (numpy arrays / lists) when
# none is supplied. CD-quality; override via the ``sample_rate`` argument.
DEFAULT_SAMPLE_RATE = 44100

__all__ = [
    "to_audio_segment",
    "to_wav_source",
    "to_wav_bytes",
    "to_waveform",
    "export_audio",
    "to_text",
    "audio_graph",
    "DEFAULT_SAMPLE_RATE",
]


# ---------------------------------------------------------------------------
# Kind predicates (recognisers)
# ---------------------------------------------------------------------------
def _is_waveform_sr(x: object) -> bool:
    """True if ``x`` is a ``(samples, sample_rate)`` pair.

    >>> import numpy as np
    >>> _is_waveform_sr((np.zeros(3), 8000))
    True
    >>> _is_waveform_sr(([0.0, 1.0], 8000))
    True
    >>> _is_waveform_sr((8000, np.zeros(3)))  # wrong order
    False
    >>> _is_waveform_sr(np.zeros(3))
    False
    """
    if not (isinstance(x, tuple) and len(x) == 2):
        return False
    samples, sr = x
    is_samples = isinstance(samples, (np.ndarray, list))
    is_sr = isinstance(sr, (int, np.integer)) and not isinstance(sr, bool)
    return bool(is_samples and is_sr)


def _is_file_like(x: object) -> bool:
    """True if ``x`` is a readable file-like object (and not bytes/str)."""
    return hasattr(x, "read") and not isinstance(x, (str, bytes, bytearray))


def sniff_audio_format(head: bytes) -> str | None:
    """Guess an audio container format from a file's leading bytes (no ffprobe).

    Used to decode WAV without shelling out to ffprobe, and shared with
    :mod:`su.cover_art` for detecting the format of raw audio bytes.

    >>> sniff_audio_format(b'ID3\\x04\\x00')
    'mp3'
    >>> sniff_audio_format(b'fLaC\\x00\\x00')
    'flac'
    >>> sniff_audio_format(b'RIFF....WAVEfmt ')
    'wav'
    >>> sniff_audio_format(b'FORM....AIFF')
    'aiff'
    >>> sniff_audio_format(b'OggS\\x00\\x02')
    'ogg'
    >>> sniff_audio_format(b'\\x00\\x00\\x00 ftypM4A ')
    'm4a'
    >>> sniff_audio_format(b'not audio') is None
    True
    """
    if head[:3] == b"ID3" or (
        len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0
    ):
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    return None


# ---------------------------------------------------------------------------
# Core conversion helpers (single source of truth; reused by graph edges)
# ---------------------------------------------------------------------------
def _pcm_from_ndarray(arr: np.ndarray) -> tuple[bytes, int]:
    """Return ``(pcm_bytes, sample_width)`` for a 1-D interleaved sample array.

    Floating-point arrays are assumed to be in ``[-1.0, 1.0]`` and are scaled to
    signed 16-bit PCM. Integer arrays keep their width (int16/int32; uint8 stays
    unsigned 8-bit, matching pydub's convention).
    """
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype("<i2").tobytes(), 2
    if arr.dtype == np.uint8:
        return arr.tobytes(), 1
    if arr.dtype in (np.int16, np.dtype("<i2"), np.dtype(">i2")):
        return arr.astype("<i2").tobytes(), 2
    if arr.dtype in (np.int32, np.dtype("<i4"), np.dtype(">i4")):
        return arr.astype("<i4").tobytes(), 4
    # Fallback: coerce anything else to int16.
    return arr.astype("<i2").tobytes(), 2


def _ndarray_to_segment(arr: np.ndarray, ctx: dict | None) -> AudioSegment:
    """Build an :class:`AudioSegment` from a numpy waveform.

    Context keys (all optional): ``sample_rate``, ``channels``, ``sample_width``.
    2-D arrays are treated as ``(num_samples, num_channels)`` and interleaved.
    """
    ctx = ctx or {}
    arr = np.asarray(arr)
    channels = ctx.get("channels")
    if arr.ndim == 2:
        # (num_samples, num_channels) -> interleaved 1-D (row-major flatten).
        channels = channels or arr.shape[1]
        arr = np.ascontiguousarray(arr).reshape(-1)
    channels = int(channels or 1)

    pcm, inferred_width = _pcm_from_ndarray(arr)
    sample_width = int(ctx.get("sample_width") or inferred_width)
    sample_rate = int(ctx.get("sample_rate") or DEFAULT_SAMPLE_RATE)
    return AudioSegment(
        data=pcm,
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=channels,
    )


def _segment_to_ndarray(
    seg: AudioSegment, *, dtype: str = "float32", mono: bool = True
) -> tuple[np.ndarray, int]:
    """Return ``(samples, sample_rate)`` from an :class:`AudioSegment`.

    Float dtypes are normalised to ``[-1.0, 1.0]``; integer dtypes keep the raw
    PCM values. Stereo is down-mixed to mono unless ``mono=False`` (in which case
    a 2-D ``(num_samples, num_channels)`` array is returned).
    """
    if mono and seg.channels > 1:
        seg = seg.set_channels(1)
    samples = np.array(seg.get_array_of_samples())
    if seg.channels > 1 and not mono:
        samples = samples.reshape((-1, seg.channels))
    if np.dtype(dtype).kind == "f":
        max_val = float(1 << (8 * seg.sample_width - 1))
        samples = samples.astype(dtype) / max_val
    else:
        samples = samples.astype(dtype)
    return samples, seg.frame_rate


def _segment_to_wav_source(seg: AudioSegment) -> io.BytesIO:
    """Export an :class:`AudioSegment` to a rewound WAV :class:`io.BytesIO`."""
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# The transformation graph: input kinds -> hub -> output kinds
# ---------------------------------------------------------------------------
audio_graph = TransformationGraph()

# Hub + input-kind nodes (with recognisers for auto-detection).
audio_graph.add_node(AudioSegment)  # isinstance-based
audio_graph.add_node("audio_path", isa=lambda x: isinstance(x, (str, os.PathLike)))
audio_graph.add_node("audio_bytes", isa=lambda x: isinstance(x, (bytes, bytearray)))
audio_graph.add_node("audio_file_like", isa=_is_file_like)
audio_graph.add_node("waveform_sr", isa=_is_waveform_sr)
audio_graph.add_node("waveform", isa=lambda x: isinstance(x, np.ndarray))
audio_graph.add_node("sample_list", isa=lambda x: isinstance(x, list))
# Output-only kinds (no recogniser: never auto-detected as an input).
audio_graph.add_node("wav_source")
audio_graph.add_node("wav_bytes")


def _from_encoded_bytes(data: bytes) -> AudioSegment:
    """Decode encoded audio bytes, avoiding ffprobe for WAV.

    pydub's ``AudioSegment.from_file`` shells out to ffprobe to detect the format
    when none is given — even for WAV, which needs no ffmpeg to decode. We sniff
    the container from magic bytes and pass ``format='wav'`` for WAV so pydub uses
    its pure-Python reader; other formats fall through to pydub's own detection
    (which needs ffmpeg anyway).
    """
    if sniff_audio_format(data[:16]) == "wav":
        return AudioSegment.from_file(io.BytesIO(data), format="wav")
    return AudioSegment.from_file(io.BytesIO(data))


@audio_graph.register_edge("audio_path", AudioSegment)
def _path_to_segment(path, ctx):
    path = os.fspath(path)
    with open(path, "rb") as fh:
        head = fh.read(16)
    if sniff_audio_format(head) == "wav":
        return AudioSegment.from_file(path, format="wav")
    return AudioSegment.from_file(path)


@audio_graph.register_edge("audio_bytes", AudioSegment)
def _bytes_to_segment(data, ctx):
    return _from_encoded_bytes(bytes(data))


@audio_graph.register_edge("audio_file_like", AudioSegment)
def _file_like_to_segment(f, ctx):
    try:
        f.seek(0)
    except Exception:
        pass
    data = f.read()
    if isinstance(data, str):
        data = data.encode("latin-1")
    return _from_encoded_bytes(data)


@audio_graph.register_edge("waveform", AudioSegment)
def _waveform_to_segment(arr, ctx):
    return _ndarray_to_segment(arr, ctx)


@audio_graph.register_edge("sample_list", AudioSegment)
def _sample_list_to_segment(lst, ctx):
    return _ndarray_to_segment(np.asarray(lst), ctx)


@audio_graph.register_edge("waveform_sr", AudioSegment)
def _waveform_sr_to_segment(pair, ctx):
    samples, sr = pair
    merged = dict(ctx or {})
    merged["sample_rate"] = int(sr)
    return _ndarray_to_segment(np.asarray(samples), merged)


@audio_graph.register_edge(AudioSegment, "wav_source")
def _seg_to_wav_source_edge(seg, ctx):
    return _segment_to_wav_source(seg)


@audio_graph.register_edge(AudioSegment, "wav_bytes")
def _seg_to_wav_bytes_edge(seg, ctx):
    return _segment_to_wav_source(seg).read()


@audio_graph.register_edge(AudioSegment, "waveform_sr")
def _seg_to_waveform_sr_edge(seg, ctx):
    ctx = ctx or {}
    return _segment_to_ndarray(
        seg, dtype=ctx.get("dtype", "float32"), mono=ctx.get("mono", True)
    )


# ---------------------------------------------------------------------------
# Public casting API (the strict normalisers)
# ---------------------------------------------------------------------------
#: Human-readable description of accepted audio inputs, for error messages.
_SUPPORTED_INPUTS = (
    "a file path (str/Path), encoded audio bytes, a binary file-like object, "
    "a numpy waveform, a (waveform, sample_rate) pair, a list of samples, "
    "or a pydub.AudioSegment"
)


def to_audio_segment(
    src,
    *,
    sample_rate: int | None = None,
    sample_width: int | None = None,
    channels: int | None = None,
) -> AudioSegment:
    """Cast any supported audio representation to a :class:`pydub.AudioSegment`.

    This is the universal ingress (Postel's "be liberal in what you accept").

    Args:
        src: The audio source. See :data:`_SUPPORTED_INPUTS`.
        sample_rate: Sample rate to *assume* for raw-sample inputs (numpy arrays
            / lists). Ignored for encoded inputs, which carry their own rate.
        sample_width: Bytes per sample for raw-sample inputs (inferred if None).
        channels: Channel count for raw-sample inputs (inferred/defaulted to 1).

    Returns:
        The audio as an :class:`AudioSegment`.

    Raises:
        TypeError: If ``src`` is not a supported audio representation.

    Example:
        >>> import numpy as np
        >>> seg = to_audio_segment(np.zeros(1000, dtype=np.int16), sample_rate=16000)
        >>> seg.frame_rate
        16000
        >>> to_audio_segment(seg) is seg  # AudioSegment passes through unchanged
        True
    """
    if isinstance(src, AudioSegment):
        return src
    context = {
        k: v
        for k, v in (
            ("sample_rate", sample_rate),
            ("sample_width", sample_width),
            ("channels", channels),
        )
        if v is not None
    }
    try:
        return audio_graph.transform_any(src, AudioSegment, context=context)
    except ConversionError as e:
        raise TypeError(
            f"Unsupported audio source of type {type(src).__name__!r}. "
            f"Expected {_SUPPORTED_INPUTS}."
        ) from e


def to_wav_source(src, **kwargs) -> io.BytesIO:
    """Cast ``src`` to a rewound WAV :class:`io.BytesIO` (for speech_recognition).

    Extra keyword arguments are forwarded to :func:`to_audio_segment`.

    Example:
        >>> import numpy as np
        >>> buf = to_wav_source(np.zeros(800, dtype=np.int16), sample_rate=8000)
        >>> buf.read(4)
        b'RIFF'
    """
    return _segment_to_wav_source(to_audio_segment(src, **kwargs))


def to_wav_bytes(src, **kwargs) -> bytes:
    """Cast ``src`` to WAV-encoded :class:`bytes`.

    Extra keyword arguments are forwarded to :func:`to_audio_segment`.

    Example:
        >>> import numpy as np
        >>> to_wav_bytes(np.zeros(800, dtype=np.int16), sample_rate=8000)[:4]
        b'RIFF'
    """
    return to_wav_source(src, **kwargs).read()


def to_waveform(
    src,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
    dtype: str = "float32",
) -> tuple[np.ndarray, int]:
    """Cast ``src`` to a ``(samples, sample_rate)`` pair for DSP/ML.

    Args:
        src: Any supported audio source.
        sample_rate: Desired sample rate of the returned waveform. For encoded
            inputs the audio is resampled to it; for raw-sample inputs it is the
            assumed rate. If None, the source's native rate is kept.
        mono: Down-mix to a single channel (returns a 1-D array). If False, a
            2-D ``(num_samples, num_channels)`` array is returned.
        dtype: Output dtype. Float dtypes are normalised to ``[-1.0, 1.0]``;
            integer dtypes keep the raw PCM values.

    Returns:
        ``(samples, sample_rate)``.

    Example:
        >>> import numpy as np
        >>> wf, sr = to_waveform(np.ones(1000, dtype=np.int16), sample_rate=16000)
        >>> wf.shape, sr, wf.dtype
        ((1000,), 16000, dtype('float32'))
    """
    seg = to_audio_segment(src, sample_rate=sample_rate)
    if sample_rate is not None and seg.frame_rate != sample_rate:
        seg = seg.set_frame_rate(sample_rate)
    return _segment_to_ndarray(seg, dtype=dtype, mono=mono)


def export_audio(
    src,
    output: str | os.PathLike | None = None,
    *,
    format: str | None = None,
    sample_rate: int | None = None,
    bitrate: str | None = None,
    tags: dict | None = None,
    **export_kwargs,
):
    """Egress: encode ``src`` to a chosen format, returning bytes or a path.

    This is the controllable-output half of the casting layer. Where the output
    format matters, pass ``format`` (or an ``output`` path with an extension);
    where it does not, omit ``output`` to get raw bytes you can pipe.

    Args:
        src: Any supported audio source.
        output: Destination path. If None, encoded bytes are returned instead.
        format: Target format (``'wav'``, ``'mp3'``, ``'flac'``, ...). Defaults to
            the ``output`` path's extension, else ``'wav'``.
        sample_rate: Optional resample target for raw-sample inputs.
        bitrate: Optional encoder bitrate (e.g. ``'192k'`` for mp3).
        tags: Optional metadata tags passed to pydub's exporter.
        **export_kwargs: Forwarded to :meth:`AudioSegment.export`.

    Returns:
        ``bytes`` if ``output`` is None, else the :class:`~pathlib.Path` written.

    Example:
        >>> import numpy as np
        >>> data = export_audio(np.zeros(800, dtype=np.int16), sample_rate=8000)
        >>> data[:4]  # WAV by default
        b'RIFF'
    """
    seg = to_audio_segment(src, sample_rate=sample_rate)
    if format is None:
        suffix = Path(os.fspath(output)).suffix if output is not None else ""
        format = suffix.lstrip(".").lower() or "wav"
    if output is None:
        buf = io.BytesIO()
        seg.export(buf, format=format, bitrate=bitrate, tags=tags, **export_kwargs)
        buf.seek(0)
        return buf.read()
    out = Path(os.fspath(output))
    out.parent.mkdir(parents=True, exist_ok=True)
    seg.export(
        os.fspath(out), format=format, bitrate=bitrate, tags=tags, **export_kwargs
    )
    return out


# ---------------------------------------------------------------------------
# Text ingress (kept simple; audio is where the graph earns its keep)
# ---------------------------------------------------------------------------
def to_text(text_src) -> str:
    """Resolve a flexible text source to a plain ``str``.

    Args:
        text_src: One of:
            - ``str``: direct text, or a file path if it looks like an absolute
              path (starts with the OS path separator or a Windows drive prefix).
            - :class:`~pathlib.Path`: path to a UTF-8 text file.
            - file-like: an object with ``.read()`` (text or binary).
            - iterable of ``str``/``bytes`` chunks.

    Returns:
        The resolved text.

    Raises:
        ValueError: If a referenced file cannot be read.
        TypeError: If ``text_src`` is an unsupported type.

    Example:
        >>> to_text("hello world")
        'hello world'
        >>> from io import StringIO
        >>> to_text(StringIO("streamed text"))
        'streamed text'
        >>> to_text(iter(["a", "b", "c"]))
        'abc'
    """
    if isinstance(text_src, str):
        looks_like_path = text_src.startswith(os.path.sep) or (
            len(text_src) > 1 and text_src[1] == ":"
        )
        if looks_like_path:
            try:
                with open(text_src, encoding="utf-8") as f:
                    return f.read()
            except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to read text file '{text_src}': {e}") from e
        return text_src

    if isinstance(text_src, Path):
        try:
            return text_src.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to read text file '{text_src}': {e}") from e

    if hasattr(text_src, "read"):
        try:
            if hasattr(text_src, "seek"):
                try:
                    text_src.seek(0)
                except (io.UnsupportedOperation, OSError):
                    pass
            content = text_src.read()
            return content.decode("utf-8") if isinstance(content, bytes) else content
        except (UnicodeDecodeError, AttributeError) as e:
            raise ValueError(f"Failed to read from text stream: {e}") from e

    if hasattr(text_src, "__iter__") and not isinstance(text_src, (str, bytes)):
        chunks = []
        for chunk in text_src:
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif isinstance(chunk, bytes):
                chunks.append(chunk.decode("utf-8"))
            else:
                raise ValueError(
                    f"Text iterator must yield str or bytes, got {type(chunk)}"
                )
        return "".join(chunks)

    raise TypeError(
        f"Unsupported text source of type {type(text_src).__name__!r}. "
        f"Expected str, Path, a file-like object, or an iterable of str/bytes."
    )
