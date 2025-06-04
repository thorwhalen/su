# su - Speech Utils

A comprehensive toolkit for speech recognition, text-to-speech generation, and audio processing with simple, intuitive interfaces.

## Installation

```bash
pip install su
```

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

- **speech_recognition**: Speech recognition functionality
- **pyttsx3**: Text-to-speech conversion
- **librosa**: Audio analysis and feature extraction
- **pydub**: Audio format conversion
- **numpy**: Numerical operations
- **pyaudio**: Audio I/O operations

## System Requirements

### For Speech Recognition:
- **Windows**: No additional requirements
- **macOS**: No additional requirements  
- **Linux**: `sudo apt-get install flac` (for FLAC support)

### For Audio Processing:
- **FFmpeg** (for format conversion): Download from https://ffmpeg.org/

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
