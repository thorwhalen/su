# su.cover_art

Embed cover images into audio files.

The headline tool, [`embed_image()`](#su.cover_art.embed_image), takes some *audio* and an *image* — each
given as a file (of various formats) or as a Python object — and produces an
audio file with the image embedded as cover art, for the formats that support it
(mp3, flac, m4a/mp4, aiff, ogg/opus).

Behaviour (all parametrizable):

- **Output format** follows the input format when that format supports
  embedding; otherwise (or when there is no file format, e.g. a numpy waveform)
  it falls back to **flac**. An explicit `output_format` (or an `output` path
  with an extension) overrides this.
- **The original file is never overwritten by default.** Set `overwrite=True`
  to replace it in place (only valid when the output format matches the input’s).
- **Output path** is either the explicit `output` you pass, a name derived from
  the input path (`<stem>_with_image.<ext>` next to it), or — when the input is
  a Python object with no path — a file in `output_dir` (default: the system
  Downloads folder).

When the source is already a file in the target format, its encoded bytes are
copied verbatim (no lossy re-encode) before the image is written in.

#### NOTE
Re-encoding a source into a different format uses **ffmpeg** via pydub. The
metadata write itself uses `mutagen` and needs no ffmpeg.

### Example

```pycon
>>> from su.cover_art import EMBEDDABLE_FORMATS
>>> sorted(EMBEDDABLE_FORMATS)
['aiff', 'flac', 'm4a', 'mp3', 'ogg', 'opus']
```

### Module Attributes

| [`EMBEDDABLE_FORMATS`](#su.cover_art.EMBEDDABLE_FORMATS)   | The set of audio formats [`embed_image()`](#su.cover_art.embed_image) can embed images into.   |
|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|

### Functions

| [`embed_image`](#su.cover_art.embed_image)(audio, image, \*[, output, ...])   | Create an audio file with an embedded cover image.   |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------|

### su.cover_art.EMBEDDABLE_FORMATS *= frozenset({'aiff', 'flac', 'm4a', 'mp3', 'ogg', 'opus'})*

The set of audio formats [`embed_image()`](#su.cover_art.embed_image) can embed images into.

### su.cover_art.embed_image(audio, image, , output=None, output_format=None, overwrite=False, output_dir=None, description='Cover')

Create an audio file with an embedded cover image.

* **Parameters:**
  * **audio** – The audio source — a file path, encoded bytes, a file-like object,
    a numpy waveform / `(waveform, sample_rate)` pair, or a
    `pydub.AudioSegment` (anything [`su.casting.to_audio_segment()`](su.casting.html.md#su.casting.to_audio_segment)
    accepts).
  * **image** – The cover image — a path, raw image bytes, a file-like object, or a
    PIL `Image`.
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit output path. If it has no extension, one is added for the
    resolved format. If None, the path is derived (see module docstring).
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Force the output format (e.g. `'mp3'`, `'flac'`,
    `'m4a'`, `'aiff'`, `'ogg'`, `'opus'`). Overrides the
    input-format-follows rule.
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, replace the original file in place (only when the
    output format matches the input’s). Defaults to False — the original
    is never touched.
  * **output_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Directory for derived output paths. Defaults to the input
    file’s directory, or the system Downloads folder for object inputs.
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The cover-art description/title stored in the tag.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) of the written audio file.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the resolved format cannot embed images, or if
      `overwrite` conflicts with the resolved output.

### Example

```pycon
>>> # File in, cover embedded, format preserved (mp3 stays mp3):
>>> embed_image("song.mp3", "cover.jpg")
PosixPath('song_with_image.mp3')
>>> # Object in (no format) -> flac in Downloads:
>>> import numpy as np
>>> embed_image(np.zeros(16000, dtype=np.int16), "cover.png")
PosixPath('/Users/you/Downloads/audio_with_image.flac')
>>> # Replace the original in place:
>>> embed_image("song.flac", img, overwrite=True)
PosixPath('song.flac')
```
