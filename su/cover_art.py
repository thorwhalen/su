"""Embed cover images into audio files.

The headline tool, :func:`embed_image`, takes some *audio* and an *image* — each
given as a file (of various formats) or as a Python object — and produces an
audio file with the image embedded as cover art, for the formats that support it
(mp3, flac, m4a/mp4, aiff, ogg/opus).

Behaviour (all parametrizable):

- **Output format** follows the input format when that format supports
  embedding; otherwise (or when there is no file format, e.g. a numpy waveform)
  it falls back to **flac**. An explicit ``output_format`` (or an ``output`` path
  with an extension) overrides this.
- **The original file is never overwritten by default.** Set ``overwrite=True``
  to replace it in place (only valid when the output format matches the input's).
- **Output path** is either the explicit ``output`` you pass, a name derived from
  the input path (``<stem>_with_image.<ext>`` next to it), or — when the input is
  a Python object with no path — a file in ``output_dir`` (default: the system
  Downloads folder).

When the source is already a file in the target format, its encoded bytes are
copied verbatim (no lossy re-encode) before the image is written in.

Note:
    Re-encoding a source into a different format uses **ffmpeg** via pydub. The
    metadata write itself uses :mod:`mutagen` and needs no ffmpeg.

Example:
    >>> from su.cover_art import EMBEDDABLE_FORMATS
    >>> sorted(EMBEDDABLE_FORMATS)
    ['aiff', 'flac', 'm4a', 'mp3', 'ogg', 'opus']
"""

import io
import os
import shutil
from collections import namedtuple
from pathlib import Path

from PIL import Image

from su.casting import export_audio, sniff_audio_format

__all__ = ["embed_image", "EMBEDDABLE_FORMATS"]

# Canonical format aliases: many spellings collapse to one embedder key.
_FORMAT_ALIASES = {
    "aif": "aiff",
    "aifc": "aiff",
    "mp4": "m4a",
    "m4b": "m4a",
    "oga": "ogg",
}

# ffmpeg/pydub export format name for each canonical format (identity if absent).
_PYDUB_EXPORT_FORMAT = {"m4a": "ipod", "aiff": "aiff"}

# Bit depth per PIL image mode, for FLAC/Ogg Picture metadata.
_PIL_MODE_DEPTH = {
    "1": 1,
    "L": 8,
    "P": 8,
    "RGB": 24,
    "RGBA": 32,
    "CMYK": 32,
    "YCbCr": 24,
    "I": 32,
    "F": 32,
}

#: Resolved image, ready for embedding.
ImageInfo = namedtuple("ImageInfo", "data mime width height depth")


def _normalize_format(fmt: str | None) -> str | None:
    """Lower-case, strip a leading dot, and collapse aliases.

    >>> _normalize_format('.MP3')
    'mp3'
    >>> _normalize_format('aif')
    'aiff'
    >>> _normalize_format(None) is None
    True
    """
    if not fmt:
        return None
    f = fmt.lower().lstrip(".")
    return _FORMAT_ALIASES.get(f, f)


# ---------------------------------------------------------------------------
# Audio-format detection
# ---------------------------------------------------------------------------
def _source_format(audio) -> str | None:
    """Return the canonical format of an audio source, or None if unknown.

    Paths use their extension; bytes / file-like objects are sniffed (via
    :func:`su.casting.sniff_audio_format`); other Python objects (waveforms,
    AudioSegment, ...) have no inherent format.
    """
    if isinstance(audio, (str, os.PathLike)):
        return _normalize_format(Path(os.fspath(audio)).suffix)
    if isinstance(audio, (bytes, bytearray)):
        return _normalize_format(sniff_audio_format(bytes(audio[:16])))
    if hasattr(audio, "read"):
        try:
            pos = audio.tell()
        except Exception:
            pos = None
        head = audio.read(16)
        if pos is not None:
            try:
                audio.seek(pos)
            except Exception:
                pass
        if isinstance(head, (bytes, bytearray)):
            return _normalize_format(sniff_audio_format(bytes(head)))
    return None


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------
def _pil_to_bytes(pil: Image.Image, fmt: str) -> bytes:
    """Serialize a PIL image to bytes in ``fmt`` ('JPEG' or 'PNG')."""
    if fmt == "JPEG" and pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def _resolve_image(image) -> ImageInfo:
    """Resolve a flexible image source to an :class:`ImageInfo`.

    Accepts a path, raw image bytes, a file-like object, or a PIL ``Image``.
    JPEG/PNG inputs keep their encoding; anything else is converted to PNG.
    """
    raw: bytes | None
    if isinstance(image, Image.Image):
        pil, raw = image, None
    else:
        if isinstance(image, (str, os.PathLike)):
            raw = Path(os.fspath(image)).read_bytes()
        elif isinstance(image, (bytes, bytearray)):
            raw = bytes(image)
        elif hasattr(image, "read"):
            try:
                image.seek(0)
            except Exception:
                pass
            raw = image.read()
        else:
            raise TypeError(
                f"Unsupported image source of type {type(image).__name__!r}. "
                f"Expected a path, bytes, a file-like object, or a PIL Image."
            )
        pil = Image.open(io.BytesIO(raw))

    fmt = (pil.format or "").upper()
    if fmt in ("JPEG", "JPG"):
        mime = "image/jpeg"
        data = raw if raw is not None else _pil_to_bytes(pil, "JPEG")
    elif fmt == "PNG":
        mime = "image/png"
        data = raw if raw is not None else _pil_to_bytes(pil, "PNG")
    else:
        # Unknown/other encodings -> PNG for broad player compatibility.
        mime = "image/png"
        data = _pil_to_bytes(pil, "PNG")

    width, height = pil.size
    depth = _PIL_MODE_DEPTH.get(pil.mode, 24)
    return ImageInfo(data=data, mime=mime, width=width, height=height, depth=depth)


# ---------------------------------------------------------------------------
# Per-format embedders (strategy pattern)
# ---------------------------------------------------------------------------
def _embed_id3(path: Path, img: ImageInfo, description: str) -> None:
    """Embed cover art via an ID3 APIC frame (mp3)."""
    from mutagen.id3 import ID3, APIC, ID3NoHeaderError

    try:
        tags = ID3(os.fspath(path))
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("APIC")
    tags.add(APIC(encoding=3, mime=img.mime, type=3, desc=description, data=img.data))
    tags.save(os.fspath(path))


def _embed_aiff(path: Path, img: ImageInfo, description: str) -> None:
    """Embed cover art into an AIFF file's ID3 chunk."""
    from mutagen.aiff import AIFF
    from mutagen.id3 import APIC

    audio = AIFF(os.fspath(path))
    if audio.tags is None:
        audio.add_tags()
    audio.tags.delall("APIC")
    audio.tags.add(
        APIC(encoding=3, mime=img.mime, type=3, desc=description, data=img.data)
    )
    audio.save()


def _embed_flac(path: Path, img: ImageInfo, description: str) -> None:
    """Embed cover art into a FLAC picture block."""
    from mutagen.flac import FLAC, Picture

    audio = FLAC(os.fspath(path))
    audio.clear_pictures()
    pic = Picture()
    pic.type = 3  # front cover
    pic.mime = img.mime
    pic.desc = description
    pic.data = img.data
    pic.width, pic.height, pic.depth = img.width, img.height, img.depth
    audio.add_picture(pic)
    audio.save()


def _embed_mp4(path: Path, img: ImageInfo, description: str) -> None:
    """Embed cover art into an MP4/M4A ``covr`` atom."""
    from mutagen.mp4 import MP4, MP4Cover

    image_format = (
        MP4Cover.FORMAT_PNG if img.mime == "image/png" else MP4Cover.FORMAT_JPEG
    )
    audio = MP4(os.fspath(path))
    audio["covr"] = [MP4Cover(img.data, imageformat=image_format)]
    audio.save()


def _make_ogg_embedder(opus: bool):
    """Build an Ogg (Vorbis/Opus) embedder using a base64 picture block."""

    def _embed(path: Path, img: ImageInfo, description: str) -> None:
        import base64

        from mutagen.flac import Picture

        if opus:
            from mutagen.oggopus import OggOpus as OggType
        else:
            from mutagen.oggvorbis import OggVorbis as OggType

        pic = Picture()
        pic.type = 3
        pic.mime = img.mime
        pic.desc = description
        pic.data = img.data
        pic.width, pic.height, pic.depth = img.width, img.height, img.depth
        encoded = base64.b64encode(pic.write()).decode("ascii")
        audio = OggType(os.fspath(path))
        audio["metadata_block_picture"] = [encoded]
        audio.save()

    return _embed


# canonical format -> embedder function
_EMBEDDERS = {
    "mp3": _embed_id3,
    "aiff": _embed_aiff,
    "flac": _embed_flac,
    "m4a": _embed_mp4,
    "ogg": _make_ogg_embedder(opus=False),
    "opus": _make_ogg_embedder(opus=True),
}

#: The set of audio formats :func:`embed_image` can embed images into.
EMBEDDABLE_FORMATS = frozenset(_EMBEDDERS)

#: Default fallback format when the input format does not support embedding.
DEFAULT_FALLBACK_FORMAT = "flac"


# ---------------------------------------------------------------------------
# Output resolution
# ---------------------------------------------------------------------------
def _decide_format(input_fmt: str | None, output, output_format: str | None) -> str:
    """Decide the output format from input format + user overrides.

    Precedence: explicit ``output_format`` > ``output`` path extension >
    input format (if embeddable) > flac fallback.

    >>> _decide_format('mp3', None, None)
    'mp3'
    >>> _decide_format('wav', None, None)          # not embeddable -> fallback
    'flac'
    >>> _decide_format(None, None, None)           # object input, no format
    'flac'
    >>> _decide_format('wav', 'out.m4a', None)     # output extension wins
    'm4a'
    >>> _decide_format('mp3', None, 'flac')        # explicit override wins
    'flac'
    >>> _decide_format('mp3', 'out.aif', None)     # aliases resolve
    'aiff'
    """
    if output_format:
        return _normalize_format(output_format)
    if output is not None:
        suffix = _normalize_format(Path(os.fspath(output)).suffix)
        if suffix:
            return suffix
    normalized_input = _normalize_format(input_fmt)
    if normalized_input in EMBEDDABLE_FORMATS:
        return normalized_input
    return DEFAULT_FALLBACK_FORMAT


def _dedupe_path(path: Path) -> Path:
    """Return ``path`` or the first ``name (n).ext`` variant that is free."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _downloads_dir() -> Path:
    """The system Downloads folder (falls back to the home directory)."""
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.home()


def _decide_output_path(
    input_path: Path | None,
    fmt: str,
    output,
    overwrite: bool,
    output_dir,
) -> Path:
    """Resolve the destination path per the documented rules."""
    ext = fmt  # canonical format == file extension
    if output is not None:
        out = Path(os.fspath(output))
        if out.suffix == "":
            out = out.with_suffix(f".{ext}")
        if (
            input_path is not None
            and not overwrite
            and out.resolve() == input_path.resolve()
        ):
            raise ValueError(
                "Refusing to overwrite the original audio file. Pass "
                "overwrite=True to replace it, or choose a different output path."
            )
        return out

    if input_path is not None:
        if overwrite:
            input_fmt = _normalize_format(input_path.suffix)
            if input_fmt != fmt:
                raise ValueError(
                    f"overwrite=True requires the output format ({fmt!r}) to match "
                    f"the input file's format ({input_fmt!r}). Provide an explicit "
                    f"output path or an output_format matching the input instead."
                )
            return input_path
        dest_dir = Path(os.fspath(output_dir)) if output_dir else input_path.parent
        return _dedupe_path(dest_dir / f"{input_path.stem}_with_image.{ext}")

    dest_dir = Path(os.fspath(output_dir)) if output_dir else _downloads_dir()
    return _dedupe_path(dest_dir / f"audio_with_image.{ext}")


# ---------------------------------------------------------------------------
# The headline tool
# ---------------------------------------------------------------------------
def embed_image(
    audio,
    image,
    *,
    output: str | os.PathLike | None = None,
    output_format: str | None = None,
    overwrite: bool = False,
    output_dir: str | os.PathLike | None = None,
    description: str = "Cover",
) -> Path:
    """Create an audio file with an embedded cover image.

    Args:
        audio: The audio source — a file path, encoded bytes, a file-like object,
            a numpy waveform / ``(waveform, sample_rate)`` pair, or a
            :class:`pydub.AudioSegment` (anything :func:`su.casting.to_audio_segment`
            accepts).
        image: The cover image — a path, raw image bytes, a file-like object, or a
            PIL ``Image``.
        output: Explicit output path. If it has no extension, one is added for the
            resolved format. If None, the path is derived (see module docstring).
        output_format: Force the output format (e.g. ``'mp3'``, ``'flac'``,
            ``'m4a'``, ``'aiff'``, ``'ogg'``, ``'opus'``). Overrides the
            input-format-follows rule.
        overwrite: If True, replace the original file in place (only when the
            output format matches the input's). Defaults to False — the original
            is never touched.
        output_dir: Directory for derived output paths. Defaults to the input
            file's directory, or the system Downloads folder for object inputs.
        description: The cover-art description/title stored in the tag.

    Returns:
        The :class:`~pathlib.Path` of the written audio file.

    Raises:
        ValueError: If the resolved format cannot embed images, or if
            ``overwrite`` conflicts with the resolved output.

    Example:
        >>> # File in, cover embedded, format preserved (mp3 stays mp3):
        >>> embed_image("song.mp3", "cover.jpg")  # doctest: +SKIP
        PosixPath('song_with_image.mp3')
        >>> # Object in (no format) -> flac in Downloads:
        >>> import numpy as np
        >>> embed_image(np.zeros(16000, dtype=np.int16), "cover.png")  # doctest: +SKIP
        PosixPath('/Users/you/Downloads/audio_with_image.flac')
        >>> # Replace the original in place:
        >>> embed_image("song.flac", img, overwrite=True)  # doctest: +SKIP
        PosixPath('song.flac')
    """
    input_path = (
        Path(os.fspath(audio)) if isinstance(audio, (str, os.PathLike)) else None
    )
    input_fmt = _source_format(audio)
    fmt = _decide_format(input_fmt, output, output_format)
    if fmt not in EMBEDDABLE_FORMATS:
        raise ValueError(
            f"Cannot embed an image into {fmt!r} audio. Embeddable formats are "
            f"{sorted(EMBEDDABLE_FORMATS)}. Pass output_format to pick one."
        )

    out_path = _decide_output_path(input_path, fmt, output, overwrite, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    same_file = input_path is not None and out_path.resolve() == input_path.resolve()
    if same_file:
        # Overwrite in place: the audio is already at out_path; embed directly.
        pass
    elif input_path is not None and _normalize_format(input_path.suffix) == fmt:
        # Same format, different destination: copy encoded bytes (no re-encode).
        shutil.copyfile(input_path, out_path)
    else:
        # Different format or object input: (re-)encode to the target format.
        export_audio(audio, out_path, format=_PYDUB_EXPORT_FORMAT.get(fmt, fmt))

    img = _resolve_image(image)
    _EMBEDDERS[fmt](out_path, img, description)
    return out_path
