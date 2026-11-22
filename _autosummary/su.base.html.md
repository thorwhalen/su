# su.base

Base speech functionalities for su: speech recognition and text-to-speech.

This module keeps the *strict, explicit* speech-processing cores small and pairs
each with a *flexible* public wrapper that casts its inputs through
[`su.casting`](su.casting.html.md#module-su.casting) (Postel’s principle: liberal inputs, strict cores).

The speech features here require the optional `speech` extra
(`pip install "su[speech]"` — brings `SpeechRecognition`, `pyttsx3` and
`pyaudio`) and the `extract_features` analysis needs the `ml` extra
(`librosa`). All such imports are **lazy**, so `import su` and the
audio-casting / cover-art tools work even when those extras are absent; a precise
`MissingDependencyError` is raised only when a feature that
needs them is actually invoked.

### Functions

| [`recognize`](#su.base.recognize)([timeout, engine])                     | Quick speech recognition from the microphone (needs the `speech` extra).   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`speak`](#su.base.speak)(text_src, \*[, rate, volume, egress, ...]) | Quick text-to-speech with flexible input and output options.               |
| [`transcribe`](#su.base.transcribe)(audio_src, \*[, engine, debug])       | Quick transcription of audio from any supported source.                    |

### Classes

| [`SpeechRecognizer`](#su.base.SpeechRecognizer)([engine])   | Easy-to-use speech recognition interface (needs the `speech` extra).   |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`TextToSpeech`](#su.base.TextToSpeech)([rate, volume]) | Easy-to-use text-to-speech interface (needs the `speech` extra).       |
| [`AudioProcessor`](#su.base.AudioProcessor)()             | Audio file processing utilities.                                       |

### *class* su.base.AudioProcessor

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

### *class* su.base.SpeechRecognizer(engine='google')

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

### *class* su.base.TextToSpeech(rate=200, volume=0.9)

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

### su.base.recognize(timeout=5, , engine='google')

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

### su.base.speak(text_src, , rate=200, volume=0.9, egress=None, send_to_speakers=True, output_format=None)

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

### su.base.transcribe(audio_src, , engine='google', debug=False)

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
