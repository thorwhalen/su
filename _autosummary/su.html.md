# su

Speech Utils (su): tools for audio I/O, speech recognition, TTS, and cover art.

The package is organised in three layers:

- [`su.casting`](su.casting.html.md#module-su.casting) — flexible input/output casting for audio (Postel’s
  principle): normalise paths, bytes, file-like objects, numpy waveforms, sample
  lists, or `AudioSegment` into the strict types the base functions need, and
  control the output format when it matters.
- [`su.base`](su.base.html.md#module-su.base) — speech recognition and text-to-speech (needs the `speech`
  extra) plus lightweight audio processing.
- [`su.cover_art`](su.cover_art.html.md#module-su.cover_art) — embed cover images into audio files.

The lightweight core (audio conversion + cover art) works out of the box.
Heavier capabilities live behind extras: `pip install "su[speech]"` (speech
recognition + TTS) and `pip install "su[ml]"` (librosa feature extraction).
Those imports are lazy, so `import su` never fails when an extra is absent.

### Example

```pycon
>>> import su
>>> import numpy as np
>>> seg = su.to_audio_segment(np.zeros(8000, dtype=np.int16), sample_rate=8000)
>>> su.to_wav_bytes(seg)[:4]
b'RIFF'
```

### Functions

| [`recognize`](#su.recognize)([timeout, engine])                     | Quick speech recognition from the microphone (needs the `speech` extra).                                                   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| [`speak`](#su.speak)(text_src, \*[, rate, volume, egress, ...]) | Quick text-to-speech with flexible input and output options.                                                               |
| [`transcribe`](#su.transcribe)(audio_src, \*[, engine, debug])       | Quick transcription of audio from any supported source.                                                                    |
| [`transcribe_words`](#su.transcribe_words)(audio, \*[, decode, ...])       | Transcribe `audio` into words with times.                                                                                  |
| [`to_audio_segment`](#su.to_audio_segment)(src, \*[, sample_rate, ...])    | Cast any supported audio representation to a `pydub.AudioSegment`.                                                         |
| [`to_wav_source`](#su.to_wav_source)(src, \*\*kwargs)                   | Cast `src` to a rewound WAV [`io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) (for speech_recognition). |
| [`to_wav_bytes`](#su.to_wav_bytes)(src, \*\*kwargs)                    | Cast `src` to WAV-encoded [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes).                               |
| [`to_waveform`](#su.to_waveform)(src, \*[, sample_rate, mono, dtype]) | Cast `src` to a `(samples, sample_rate)` pair for DSP/ML.                                                                  |
| [`export_audio`](#su.export_audio)(src[, output, format, ...])         | Egress: encode `src` to a chosen format, returning bytes or a path.                                                        |
| [`to_text`](#su.to_text)(text_src)                                | Resolve a flexible text source to a plain `str`.                                                                           |
| [`embed_image`](#su.embed_image)(audio, image, \*[, output, ...])     | Create an audio file with an embedded cover image.                                                                         |

### Classes

| [`SpeechRecognizer`](#su.SpeechRecognizer)([engine])                      | Easy-to-use speech recognition interface (needs the `speech` extra).   |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`TextToSpeech`](#su.TextToSpeech)([rate, volume])                    | Easy-to-use text-to-speech interface (needs the `speech` extra).       |
| [`AudioProcessor`](#su.AudioProcessor)()                                | Audio file processing utilities.                                       |
| [`WordTranscript`](#su.WordTranscript)(segments[, speech_regions, ...]) | The transcript: segments with timed words, and what to be wary of.     |

### *class* su.AudioProcessor

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Audio file processing utilities.

`load_audio` and `convert_format` use the lightweight core (pydub);
`extract_features` needs the `ml` extra (librosa).

#### *static* convert_format(input_path, output_path, format='wav')

Convert an audio file from one format to another.

* **Parameters:**
  * **input_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Input file path (or any supported audio source).
  * **output_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Output file path.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Target format (wav, mp3, flac, …). Non-WAV needs ffmpeg.
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if conversion succeeded, else False.

### Example

```pycon
>>> AudioProcessor.convert_format("input.mp3", "output.wav")
True
```

#### *static* extract_features(audio_data, sample_rate)

Extract basic audio features for ML (needs the `ml` extra).

* **Parameters:**
  * **audio_data** (`ndarray`) – Audio time series.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary with MFCC, spectral, tempo and beat features.

### Example

```pycon
>>> audio, sr = AudioProcessor.load_audio("speech.wav")
>>> features = AudioProcessor.extract_features(audio, sr)
```

#### *static* load_audio(file_path, sample_rate=None)

Load audio to a `(waveform, sample_rate)` pair.

Accepts any source [`su.casting.to_waveform()`](su.casting.html.md#su.casting.to_waveform) accepts. Uses the
lightweight core (no librosa required).

* **Parameters:**
  * **file_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO)) – Audio source (path, bytes, file-like, waveform, …).
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Target sample rate (None keeps the native rate).
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Tuple of (audio_data, sample_rate), audio as float32 mono in [-1, 1].

### Example

```pycon
>>> import numpy as np
>>> from su.casting import to_wav_bytes
>>> wav = to_wav_bytes(np.zeros(8000, dtype=np.int16), sample_rate=8000)
>>> audio, sr = AudioProcessor.load_audio(wav)
>>> audio.dtype, sr
(dtype('float32'), 8000)
```

### *class* su.SpeechRecognizer(engine='google')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Easy-to-use speech recognition interface (needs the `speech` extra).

#### listen_and_recognize(timeout=5)

Listen to the microphone and return recognized text.

* **Parameters:**
  **timeout** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum seconds to wait for speech.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Recognized text or None if recognition failed.

### Example

```pycon
>>> recognizer = SpeechRecognizer()
>>> text = recognizer.listen_and_recognize()
```

#### *property* microphone

The microphone source, created lazily (needs `pyaudio`).

#### recognize_file(audio_file)

Recognize speech from an audio source of any supported form.

* **Parameters:**
  **audio_file** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO)) – A file path, bytes, or file-like object. Non-WAV formats
  are transcoded to WAV first (needs ffmpeg).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Recognized text or None if recognition failed.

### Example

```pycon
>>> recognizer = SpeechRecognizer()
>>> text = recognizer.recognize_file("speech.wav")
```

### *class* su.TextToSpeech(rate=200, volume=0.9)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Easy-to-use text-to-speech interface (needs the `speech` extra).

#### get_voices()

Get available voices.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### set_voice(voice_id)

Set voice by ID.

#### speak(text, save_to=None)

Convert text to speech.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to convert to speech.
  * **save_to** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional file path to save audio.

### Example

```pycon
>>> tts = TextToSpeech()
>>> tts.speak("Hello, world!")
```

### *class* su.WordTranscript(segments, speech_regions=(), untranscribed=(), dropped=(), separated=False, aligned=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The transcript: segments with timed words, and what to be wary of.

#### aligned *: [bool](https://docs.python.org/3/builtins/functions.html#bool)* *= False*

Whether at least one word got aligned times.

#### dropped *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Segment](su.words.html.md#su.words.Segment), ...]* *= ()*

Segments dropped as known hallucinations.

#### *property* text *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The kept text, one string.

#### untranscribed *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), [float](https://docs.python.org/3/builtins/functions.html#float)], ...]* *= ()*

Speech regions (or parts of them) that no kept word covers.

#### *property* words *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[Word](su.words.html.md#su.words.Word)]*

Every word, in order.

### su.embed_image(audio, image, , output=None, output_format=None, overwrite=False, output_dir=None, description='Cover')

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

### su.export_audio(src, output=None, , format=None, sample_rate=None, bitrate=None, tags=None, \*\*export_kwargs)

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

### su.recognize(timeout=5, , engine='google')

Quick speech recognition from the microphone (needs the `speech` extra).

* **Parameters:**
  * **timeout** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum seconds to wait for speech.
  * **engine** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Recognition engine (‘google’, ‘sphinx’, ‘wit’, ‘azure’, ‘houndify’).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Example

```pycon
>>> text = recognize()
```

### su.speak(text_src, , rate=200, volume=0.9, egress=None, send_to_speakers=True, output_format=None)

Quick text-to-speech with flexible input and output options.

* **Parameters:**
  * **text_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Any source accepted by [`su.casting.to_text()`](su.casting.html.md#su.casting.to_text) (direct text,
    a text-file path/Path, a file-like object, or an iterable of chunks).
  * **rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Speech rate (words per minute).
  * **volume** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Volume level (0.0 to 1.0).
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    Optional output handler:
    - None: no file/bytes output (just speakers, if enabled).
    - str/Path: file path to save audio; the extension (or
      `output_format`) selects the encoding.
    - Callable: receives the synthesized WAV bytes and its return value
      is returned.
  * **send_to_speakers** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to also play the audio through the speakers.
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit output format for a path `egress` (overrides the
    extension). Where format does not matter, use a callable `egress` to
    pipe the raw bytes.
* **Returns:**
  The egress callable’s result, the written [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), or None.

### Example

```pycon
>>> speak("Hello, world!")
>>> speak("Save this", egress="out.mp3", send_to_speakers=False)
>>> audio = speak("Test", egress=lambda b: b, send_to_speakers=False)
```

### su.to_audio_segment(src, , sample_rate=None, sample_width=None, channels=None)

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

### su.to_text(text_src)

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

### su.to_wav_bytes(src, \*\*kwargs)

Cast `src` to WAV-encoded [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes).

Extra keyword arguments are forwarded to [`to_audio_segment()`](#su.to_audio_segment).

### Example

```pycon
>>> import numpy as np
>>> to_wav_bytes(np.zeros(800, dtype=np.int16), sample_rate=8000)[:4]
b'RIFF'
```

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### su.to_wav_source(src, \*\*kwargs)

Cast `src` to a rewound WAV [`io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) (for speech_recognition).

Extra keyword arguments are forwarded to [`to_audio_segment()`](#su.to_audio_segment).

### Example

```pycon
>>> import numpy as np
>>> buf = to_wav_source(np.zeros(800, dtype=np.int16), sample_rate=8000)
>>> buf.read(4)
b'RIFF'
```

* **Return type:**
  [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO)

### su.to_waveform(src, , sample_rate=None, mono=True, dtype='float32')

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

### su.transcribe(audio_src, , engine='google', debug=False)

Quick transcription of audio from any supported source.

Non-WAV inputs are transcoded to WAV via [`su.casting.to_wav_source()`](su.casting.html.md#su.casting.to_wav_source)
(needs ffmpeg) before being handed to the strict recognition core.

* **Parameters:**
  * **audio_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Any source accepted by [`su.casting.to_audio_segment()`](su.casting.html.md#su.casting.to_audio_segment),
    or a `{'type': 'microphone', 'timeout': N}` dict for live capture.
  * **engine** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Recognition engine (‘google’, ‘sphinx’, ‘wit’, ‘azure’, ‘houndify’).
  * **debug** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print debug information and re-raise unexpected errors if True.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Transcribed text, or None if transcription failed.

### Example

```pycon
>>> text = transcribe("recording.wav")
>>> text = transcribe({'type': 'microphone', 'timeout': 5})
```

### su.transcribe_words(audio, \*, decode=<function \_decode>, separate=None, detect_speech=None, recognise=None, drop=<function is_hallucination>, align=None, separation=True, alignment=True)

Transcribe `audio` into words with times.

* **Parameters:**
  * **audio** – any source [`su.casting.to_waveform()`](su.casting.html.md#su.casting.to_waveform) accepts (a path, bytes,
    a file-like, a `(samples, sample_rate)` pair).
  * **decode** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`object`](https://docs.python.org/3/builtins/functions.html#object)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]]) – audio → mono `(samples, sample_rate)`, at any rate.
  * **separate** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – keep the voice, returning 16 kHz (default `demucs_separator()`).
  * **detect_speech** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – speech regions in seconds (default `silero_speech_regions()`).
  * **recognise** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](su.words.html.md#su.words.Segment)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – segments with timed words (default `whisper_recogniser()`).
  * **drop** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – `text -> bool`, which segments to set apart as hallucinations.
  * **align** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](su.words.html.md#su.words.Segment)]], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](su.words.html.md#su.words.Segment)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – refine word times (default `mms_aligner()`).
  * **separation** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – `False` skips separation, for speech without music.
  * **alignment** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – `False` keeps Whisper’s word times.
* **Return type:**
  [`WordTranscript`](su.words.html.md#su.words.WordTranscript)

### Modules

| [`base`](su.base.html.md#module-su.base)           | Base speech functionalities for su: speech recognition and text-to-speech.   |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`casting`](su.casting.html.md#module-su.casting)     | Flexible input/output casting for audio (and text).                          |
| [`cover_art`](su.cover_art.html.md#module-su.cover_art) | Embed cover images into audio files.                                         |
| [`words`](su.words.html.md#module-su.words)         | Word-timestamped transcription of speech, including speech under music.      |
