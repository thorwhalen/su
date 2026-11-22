# su.casting

Flexible input/output casting for audio (and text).

Postel’s principle applied to audio: \*be liberal in what you accept, strict in
what the base functions require.\* This module provides a small
`i2.castgraph`-based transformation layer that normalises the many
*representations* an audio resource can take — file paths, encoded bytes,
file-like objects, numpy waveforms, plain sample lists, or a
`pydub.AudioSegment` — into the strict types the base functions consume,
and lets callers control the *output* format when it matters (and pipe raw bytes
when it does not).

## Canonical hub

`pydub.AudioSegment` is the canonical in-memory audio type (the “hub”):
it carries sample rate / channels / sample width, constructs from raw PCM or any
ffmpeg-decodable file/bytes, and exports to any format. Every *input* kind
converts **to** the hub; every *output* is derived **from** it. castgraph composes
multi-hop routes automatically (e.g. `audio_path -> AudioSegment -> wav_source`).

## Kinds

Input kinds:  `'audio_path'`, `'audio_bytes'`, `'audio_file_like'`,
`'waveform'`, `'waveform_sr'`, `'sample_list'`, and the hub
`AudioSegment`.

Output kinds: `'wav_source'`, `'wav_bytes'`, `'waveform_sr'` (+ the hub).

## Public API

- [`to_audio_segment()`](#su.casting.to_audio_segment) — universal ingress to the hub.
- [`to_wav_source()`](#su.casting.to_wav_source) / [`to_wav_bytes()`](#su.casting.to_wav_bytes) — WAV for `speech_recognition`.
- [`to_waveform()`](#su.casting.to_waveform) — `(np.ndarray, sample_rate)` for DSP/ML.
- [`export_audio()`](#su.casting.export_audio) — the egress: choose the output format, or get bytes.
- [`to_text()`](#su.casting.to_text) — flexible text ingress.
- `audio_graph` — the underlying `TransformationGraph`.

#### NOTE
Decoding/encoding non-WAV formats (mp3, flac, m4a, …) uses **ffmpeg** via
pydub. Install it separately (e.g. `brew install ffmpeg` /
`apt-get install ffmpeg`). WAV and raw-sample conversions need no ffmpeg.

### Example

```pycon
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
```

### Functions

| [`to_audio_segment`](#su.casting.to_audio_segment)(src, \*[, sample_rate, ...])    | Cast any supported audio representation to a `pydub.AudioSegment`.                                                         |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| [`to_wav_source`](#su.casting.to_wav_source)(src, \*\*kwargs)                   | Cast `src` to a rewound WAV [`io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) (for speech_recognition). |
| [`to_wav_bytes`](#su.casting.to_wav_bytes)(src, \*\*kwargs)                    | Cast `src` to WAV-encoded [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes).                               |
| [`to_waveform`](#su.casting.to_waveform)(src, \*[, sample_rate, mono, dtype]) | Cast `src` to a `(samples, sample_rate)` pair for DSP/ML.                                                                  |
| [`export_audio`](#su.casting.export_audio)(src[, output, format, ...])         | Egress: encode `src` to a chosen format, returning bytes or a path.                                                        |
| [`to_text`](#su.casting.to_text)(text_src)                                | Resolve a flexible text source to a plain `str`.                                                                           |

### su.casting.export_audio(src, output=None, , format=None, sample_rate=None, bitrate=None, tags=None, \*\*export_kwargs)

Egress: encode `src` to a chosen format, returning bytes or a path.

This is the controllable-output half of the casting layer. Where the output
format matters, pass `format` (or an `output` path with an extension);
where it does not, omit `output` to get raw bytes you can pipe.

* **Parameters:**
  * **src** – Any supported audio source.
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Destination path. If None, encoded bytes are returned instead.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Target format (`'wav'`, `'mp3'`, `'flac'`, …). Defaults to
    the `output` path’s extension, else `'wav'`.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional resample target for raw-sample inputs.
  * **bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional encoder bitrate (e.g. `'192k'` for mp3).
  * **tags** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional metadata tags passed to pydub’s exporter.
  * **\*\*export_kwargs** – Forwarded to `AudioSegment.export()`.
* **Returns:**
  `bytes` if `output` is None, else the [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) written.

### Example

```pycon
>>> import numpy as np
>>> data = export_audio(np.zeros(800, dtype=np.int16), sample_rate=8000)
>>> data[:4]  # WAV by default
b'RIFF'
```

### su.casting.to_audio_segment(src, , sample_rate=None, sample_width=None, channels=None)

Cast any supported audio representation to a `pydub.AudioSegment`.

This is the universal ingress (Postel’s “be liberal in what you accept”).

* **Parameters:**
  * **src** – The audio source. See `_SUPPORTED_INPUTS`.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sample rate to *assume* for raw-sample inputs (numpy arrays
    / lists). Ignored for encoded inputs, which carry their own rate.
  * **sample_width** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Bytes per sample for raw-sample inputs (inferred if None).
  * **channels** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Channel count for raw-sample inputs (inferred/defaulted to 1).
* **Return type:**
  `AudioSegment`
* **Returns:**
  The audio as an `AudioSegment`.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – If `src` is not a supported audio representation.

### Example

```pycon
>>> import numpy as np
>>> seg = to_audio_segment(np.zeros(1000, dtype=np.int16), sample_rate=16000)
>>> seg.frame_rate
16000
>>> to_audio_segment(seg) is seg  # AudioSegment passes through unchanged
True
```

### su.casting.to_text(text_src)

Resolve a flexible text source to a plain `str`.

* **Parameters:**
  **text_src** – 

  One of:
  - `str`: direct text, or a file path if it looks like an absolute
    path (starts with the OS path separator or a Windows drive prefix).
  - [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path): path to a UTF-8 text file.
  - file-like: an object with `.read()` (text or binary).
  - iterable of `str`/`bytes` chunks.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The resolved text.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If a referenced file cannot be read.
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – If `text_src` is an unsupported type.

### Example

```pycon
>>> to_text("hello world")
'hello world'
>>> from io import StringIO
>>> to_text(StringIO("streamed text"))
'streamed text'
>>> to_text(iter(["a", "b", "c"]))
'abc'
```

### su.casting.to_wav_bytes(src, \*\*kwargs)

Cast `src` to WAV-encoded [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes).

Extra keyword arguments are forwarded to [`to_audio_segment()`](#su.casting.to_audio_segment).

### Example

```pycon
>>> import numpy as np
>>> to_wav_bytes(np.zeros(800, dtype=np.int16), sample_rate=8000)[:4]
b'RIFF'
```

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### su.casting.to_wav_source(src, \*\*kwargs)

Cast `src` to a rewound WAV [`io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) (for speech_recognition).

Extra keyword arguments are forwarded to [`to_audio_segment()`](#su.casting.to_audio_segment).

### Example

```pycon
>>> import numpy as np
>>> buf = to_wav_source(np.zeros(800, dtype=np.int16), sample_rate=8000)
>>> buf.read(4)
b'RIFF'
```

* **Return type:**
  [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO)

### su.casting.to_waveform(src, , sample_rate=None, mono=True, dtype='float32')

Cast `src` to a `(samples, sample_rate)` pair for DSP/ML.

* **Parameters:**
  * **src** – Any supported audio source.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Desired sample rate of the returned waveform. For encoded
    inputs the audio is resampled to it; for raw-sample inputs it is the
    assumed rate. If None, the source’s native rate is kept.
  * **mono** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Down-mix to a single channel (returns a 1-D array). If False, a
    2-D `(num_samples, num_channels)` array is returned.
  * **dtype** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output dtype. Float dtypes are normalised to `[-1.0, 1.0]`;
    integer dtypes keep the raw PCM values.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  `(samples, sample_rate)`.

### Example

```pycon
>>> import numpy as np
>>> wf, sr = to_waveform(np.ones(1000, dtype=np.int16), sample_rate=16000)
>>> wf.shape, sr, wf.dtype
((1000,), 16000, dtype('float32'))
```
