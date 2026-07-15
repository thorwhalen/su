# su - Speech Utils

A comprehensive toolkit for speech recognition, text-to-speech generation, and audio processing with simple, intuitive interfaces.

## Installation

`su` has a **lightweight core** (audio format conversion, flexible audio I/O, and
cover-art embedding) that works out of the box:

```bash
pip install su
```

Heavier / more fragile capabilities are optional extras:

```bash
pip install "su[speech]"   # speech recognition + text-to-speech
pip install "su[ml]"       # librosa feature extraction (MFCC, spectral, tempo)
pip install "su[full]"     # everything
```

Extra imports are **lazy**, so `import su` never fails when an extra is missing —
you only get a precise "install `su[...]`" error if you call a feature that needs
one.

> **System dependency:** decoding/encoding non-WAV formats (mp3, flac, m4a, …)
> uses [**ffmpeg**](https://ffmpeg.org/). Install it with `brew install ffmpeg`
> (macOS), `sudo apt-get install ffmpeg` (Debian/Ubuntu), or from ffmpeg.org.
> WAV and raw-waveform conversions need no ffmpeg.

## Quick Start

### Speech Recognition

```python
import su

# Quick recognition from microphone
text = su.recognize()
print(f"You said: {text}")

# Custom timeout and engine
text = su.recognize(timeout=10, engine='sphinx')

# Transcribe from various audio sources
text = su.transcribe("recording.wav")  # File path
print(f"Audio contains: {text}")

# Transcribe from bytes
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()
text = su.transcribe(audio_bytes)

# Transcribe from file-like object
from io import BytesIO
audio_stream = BytesIO(audio_bytes)
text = su.transcribe(audio_stream)

# Transcribe from live microphone using transcribe
text = su.transcribe({'type': 'microphone', 'timeout': 5})

# Use offline engine for transcription
text = su.transcribe("recording.wav", engine='sphinx')

# Advanced usage
recognizer = su.SpeechRecognizer(engine='google')
text = recognizer.listen_and_recognize(timeout=10)
```

### Text-to-Speech

```python
import su

# Quick speech
su.speak("Hello, world!")

# Custom voice settings
su.speak("Slow and quiet", rate=100, volume=0.5)

# Read text from file (path starting with /)
su.speak("/path/to/speech.txt")

# Save to file without hearing
su.speak("Save this", egress="output.wav", send_to_speakers=False)

# Get audio bytes for custom use
audio_bytes = su.speak("Test", egress=lambda x: x, send_to_speakers=False)

# Both save and hear
su.speak("Hello", egress="greeting.wav", send_to_speakers=True)

# Advanced usage
tts = su.TextToSpeech(rate=150, volume=0.8)
tts.speak("This is a test", save_to="output.wav")

# List available voices
voices = tts.get_voices()
for voice in voices:
    print(f"Voice: {voice['name']} ({voice['lang']})")
```

### Partial Application for Custom Functions

```python
import su
from functools import partial

# Create custom recognizer functions
fast_recognize = partial(su.recognize, timeout=2, engine='google')
offline_recognize = partial(su.recognize, engine='sphinx')

# Create custom speech functions
robot_voice = partial(su.speak, rate=300, volume=1.0)
quiet_voice = partial(su.speak, rate=150, volume=0.3)

# Create custom transcription functions
offline_transcribe = partial(su.transcribe, engine='sphinx')
google_transcribe = partial(su.transcribe, engine='google')

# Use them
text = fast_recognize()  # Quick 2-second recognition
robot_voice("I am a robot")  # Fast, loud speech
text = offline_transcribe("audio.wav")  # Offline transcription
```

### Audio Processing

```python
import su

# Load and analyze audio
audio, sample_rate = su.AudioProcessor.load_audio("speech.wav")
features = su.AudioProcessor.extract_features(audio, sample_rate)

print(f"MFCC shape: {features['mfcc'].shape}")
print(f"Tempo: {features['tempo']} BPM")

# Convert audio formats
su.AudioProcessor.convert_format("input.mp3", "output.wav")
```

### 🖼️ Embed Cover Art

Give it some audio and an image — as files (of various formats) or as Python
objects — and get back an audio file with the image embedded as cover art.

```python
import su

# File in → cover embedded, format preserved (mp3 stays mp3).
# The original is NOT overwritten by default:
out = su.embed_image("song.mp3", "cover.jpg")
# -> Path("song_with_image.mp3")

# Replace the original in place when you ask for it:
su.embed_image("song.flac", "cover.png", overwrite=True)

# Choose the output format and location explicitly:
su.embed_image("song.wav", "cover.jpg", output="album/track.m4a")

# Python objects work too — a numpy waveform + a PIL image.
# With no source file format, output defaults to FLAC, written to your
# Downloads folder (override with output_dir=...):
import numpy as np
from PIL import Image
waveform = np.random.uniform(-0.2, 0.2, 16000).astype("float32")
su.embed_image(waveform, Image.open("cover.png"))
# -> Path("~/Downloads/audio_with_image.flac")
```

Embeddable formats: **mp3, flac, m4a/mp4, aiff, ogg, opus** (`su.EMBEDDABLE_FORMATS`).
When the input format can't hold cover art (e.g. wav) or there's no file format
at all, the output falls back to FLAC unless you pass `output_format`.

### 🔀 Flexible Audio I/O (Casting)

The casting layer follows Postel's principle — *be liberal in what you accept,
strict in what you emit*. Any audio representation (path, encoded bytes, a
file-like object, a numpy waveform, a `(waveform, sample_rate)` pair, a list of
samples, or a `pydub.AudioSegment`) casts to the type you need; you control the
output format when it matters.

```python
import su
import numpy as np

# A numpy waveform → an AudioSegment (the canonical in-memory hub):
seg = su.to_audio_segment(np.zeros(8000, dtype=np.int16), sample_rate=8000)

# Any source → a (waveform, sample_rate) pair for DSP/ML:
waveform, sr = su.to_waveform("song.mp3", sample_rate=16000, mono=True)

# Any source → WAV bytes (no ffmpeg needed for WAV output):
wav_bytes = su.to_wav_bytes(seg)

# Control the output format on the way out — or omit `output` to get bytes to pipe:
su.export_audio(seg, "out.flac")            # writes FLAC, returns the Path
mp3_bytes = su.export_audio(seg, format="mp3", bitrate="192k")  # returns bytes
```

Under the hood this is an [`i2.castgraph`](https://github.com/i2mint/i2)
transformation graph: every input kind routes **to** the `AudioSegment` hub and
every output is derived **from** it, with multi-hop routes composed
automatically.

## Features

### 🎤 Speech Recognition
- **Multiple Engines**: Google, Sphinx, Wit.ai, Azure, Houndify
- **Live Recognition**: Real-time microphone input
- **File Transcription**: Support for various audio formats
- **Noise Handling**: Automatic ambient noise adjustment

### 🔊 Text-to-Speech
- **Cross-Platform**: Works on Windows, macOS, Linux
- **Voice Control**: Rate, volume, and voice selection
- **File Export**: Save speech to audio files
- **Multiple Voices**: Access to system voices

### 🎵 Audio Processing
- **Format Conversion**: MP3, WAV, FLAC, and more
- **Feature Extraction**: MFCC, spectral features, tempo
- **ML Ready**: Features suitable for machine learning
- **Librosa Integration**: Advanced audio analysis

### 🖼️ Cover Art
- **Embed Images**: mp3, flac, m4a/mp4, aiff, ogg, opus
- **Format-Preserving**: keeps the input format when it supports embedding
- **Safe by Default**: never overwrites the original unless you ask
- **Flexible Inputs**: audio and images as files or Python objects

### 🔀 Flexible I/O (Casting)
- **Postel's Principle**: liberal inputs, strict, annotatable cores
- **Many Representations**: paths, bytes, file-like, numpy waveforms, AudioSegment
- **Controllable Output**: choose the output format, or pipe raw bytes
- **castgraph-Powered**: canonical `AudioSegment` hub with auto-routed conversions

## API Reference

### Convenience Functions

```python
# Speech recognition with customizable settings
text = su.recognize(timeout=5, engine='google')

# Flexible text-to-speech with multiple input/output options
result = su.speak(text_src, rate=200, volume=0.9, egress=None, send_to_speakers=True)

# Where text_src can be:
# - "Hello world" (direct text)
# - "/path/to/file.txt" (file path - must start with / or drive letter)
# - Path("file.txt") (Path object)
# - StringIO("text") (file-like object)
# - text_iterator() (iterator yielding text chunks)

# Where egress can be:
# - None (default - no special output)
# - "output.wav" (save to file path)
# - lambda x: x (return audio bytes)
# - custom_function (process audio bytes)

# Flexible audio transcription with multiple source types
text = su.transcribe(audio_src, engine='google')
```

### SpeechRecognizer

```python
recognizer = su.SpeechRecognizer(engine='google')

# Listen from microphone
text = recognizer.listen_and_recognize(timeout=5)

# Transcribe file
text = recognizer.recognize_file("audio.wav")
```

### TextToSpeech

```python
tts = su.TextToSpeech(rate=200, volume=0.9)

# Speak text
tts.speak("Hello world")

# Save to file
tts.speak("Save this", save_to="output.wav")

# Change voice
voices = tts.get_voices()
tts.set_voice(voices[0]['id'])
```

### AudioProcessor

```python
# Load audio
audio, sr = su.AudioProcessor.load_audio("file.wav")

# Extract ML features
features = su.AudioProcessor.extract_features(audio, sr)

# Convert format
su.AudioProcessor.convert_format("input.mp3", "output.wav")
```

## Dependencies

**Core** (always installed):

- **i2**: the `castgraph` transformation engine behind flexible casting
- **pydub**: audio format conversion (via ffmpeg)
- **numpy**: waveform / numerical operations
- **mutagen**: reading/writing audio metadata (cover art)
- **Pillow**: image handling for cover art

**Extras** (opt in):

- **`su[speech]`** → `SpeechRecognition`, `pyttsx3`, `pyaudio` — recognition + TTS
- **`su[ml]`** → `librosa`, `soundfile` — feature extraction / analysis

## System Requirements

### FFmpeg (for non-WAV audio)
Decoding/encoding mp3, flac, m4a, ogg, etc. uses **ffmpeg**:
- **macOS**: `brew install ffmpeg`
- **Debian/Ubuntu**: `sudo apt-get install ffmpeg`
- **Windows / other**: download from https://ffmpeg.org/

WAV and raw-waveform conversions work without ffmpeg.

### For Speech Recognition (`su[speech]`):
- **Windows / macOS**: no additional system requirements
- **Linux**: `sudo apt-get install flac portaudio19-dev` (FLAC + PortAudio for the microphone)

## Examples

### Voice Assistant with Custom Settings

```python
import su
from functools import partial

# Create optimized functions for the assistant
quick_listen = partial(su.recognize, timeout=3, engine='google')
assistant_voice = partial(su.speak, rate=180, volume=0.8)

while True:
    print("Listening...")
    text = quick_listen()
    
    if text:
        print(f"You said: {text}")
        response = f"You said: {text}"
        assistant_voice(response)
    
    if text and "goodbye" in text.lower():
        assistant_voice("Goodbye!")
        break
```

### Audio Analysis Pipeline

```python
import su
import numpy as np

# Load audio file
audio, sr = su.AudioProcessor.load_audio("speech.wav")

# Extract features for ML
features = su.AudioProcessor.extract_features(audio, sr)

# Use MFCC features (common for speech recognition)
mfcc_features = features['mfcc']
mfcc_mean = np.mean(mfcc_features, axis=1)

print(f"MFCC feature vector shape: {mfcc_mean.shape}")
```

### Batch Processing with Different Engines

```python
import su
from functools import partial
from pathlib import Path

# Create specialized transcription functions
google_transcribe = partial(su.transcribe, engine='google')  # For online processing
sphinx_transcribe = partial(su.transcribe, engine='sphinx')  # For offline processing

input_dir = Path("audio_files")
output_dir = Path("transcriptions")
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob("*.wav"):
    print(f"Processing {audio_file.name}...")
    
    # Try Google first (better accuracy), fallback to Sphinx
    text = google_transcribe(audio_file) or sphinx_transcribe(audio_file)
    
    # Save transcription
    output_file = output_dir / f"{audio_file.stem}.txt"
    with open(output_file, "w") as f:
        f.write(text or "Transcription failed")
```

### Voice Profile System

```python
import su
from functools import partial

# Define different voice profiles
profiles = {
    'assistant': partial(su.speak, rate=180, volume=0.8),
    'narrator': partial(su.speak, rate=150, volume=0.7),
    'robot': partial(su.speak, rate=250, volume=1.0),
    'whisper': partial(su.speak, rate=120, volume=0.3),
}

# Use different voices for different purposes
profiles['assistant']("How can I help you today?")
profiles['narrator']("Once upon a time, in a land far away...")
profiles['robot']("SYSTEM INITIALIZED. READY FOR COMMANDS.")
profiles['whisper']("This is a secret message.")

# Save different voice outputs
for name, voice_func in profiles.items():
    voice_func(f"This is the {name} voice.", save_to=f"{name}_sample.wav")
```

### Flexible Audio Sources

The `transcribe()` function accepts audio from multiple sources:

```python
import su
from io import BytesIO

# 1. File paths (strings or Path objects)
text = su.transcribe("recording.wav")
text = su.transcribe(Path("audio/speech.mp3"))

# 2. Raw audio bytes
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()
text = su.transcribe(audio_bytes)

# 3. File-like objects (BytesIO, open files, etc.)
audio_stream = BytesIO(audio_bytes)
text = su.transcribe(audio_stream)

# 4. Open file handles
with open("recording.wav", "rb") as f:
    text = su.transcribe(f)

# 5. Audio chunk iterators
def audio_chunks():
    with open("large_audio.wav", "rb") as f:
        while True:
            chunk = f.read(8192)  # 8KB chunks
            if not chunk:
                break
            yield chunk

text = su.transcribe(audio_chunks())

# 6. Live microphone via transcribe
text = su.transcribe({'type': 'microphone', 'timeout': 10})

# 7. Network streams or any file-like object
import requests
response = requests.get("https://example.com/audio.wav", stream=True)
text = su.transcribe(BytesIO(response.content))
```

### Batch Processing with Different Sources

```python
import su
from functools import partial
from pathlib import Path
from io import BytesIO

# Create specialized transcription functions
google_transcribe = partial(su.transcribe, engine='google')
sphinx_transcribe = partial(su.transcribe, engine='sphinx')

# Process various audio sources
sources = [
    "local_file.wav",                                    # File path
    BytesIO(audio_bytes),                               # Bytes stream
    {'type': 'microphone', 'timeout': 3},              # Live microphone
    Path("recordings/interview.mp3"),                   # Path object
]

for i, source in enumerate(sources):
    print(f"Processing source {i+1}...")
    
    # Try Google first, fallback to Sphinx
    text = google_transcribe(source) or sphinx_transcribe(source)
    
    print(f"Result: {text or 'Transcription failed'}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
