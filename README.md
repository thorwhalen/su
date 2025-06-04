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
text = su.quick_recognize()
print(f"You said: {text}")

# Transcribe audio file
text = su.quick_transcribe("recording.wav")
print(f"Audio contains: {text}")

# Advanced usage
recognizer = su.SpeechRecognizer(engine='google')
text = recognizer.listen_and_recognize(timeout=10)
```

### Text-to-Speech

```python
import su

# Quick speech
su.quick_speak("Hello, world!")

# Advanced usage
tts = su.TextToSpeech(rate=150, volume=0.8)
tts.speak("This is a test", save_to="output.wav")

# List available voices
voices = tts.get_voices()
for voice in voices:
    print(f"Voice: {voice['name']} ({voice['lang']})")
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

### Voice Assistant Basic Loop

```python
import su

recognizer = su.SpeechRecognizer()
tts = su.TextToSpeech()

while True:
    print("Listening...")
    text = recognizer.listen_and_recognize(timeout=5)
    
    if text:
        print(f"You said: {text}")
        response = f"You said: {text}"
        tts.speak(response)
    
    if text and "goodbye" in text.lower():
        tts.speak("Goodbye!")
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

### Batch Audio Processing

```python
import su
from pathlib import Path

input_dir = Path("audio_files")
output_dir = Path("processed_audio")
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob("*.mp3"):
    output_file = output_dir / f"{audio_file.stem}.wav"
    
    # Convert to WAV
    su.AudioProcessor.convert_format(audio_file, output_file)
    
    # Transcribe
    text = su.quick_transcribe(output_file)
    
    # Save transcription
    with open(output_dir / f"{audio_file.stem}.txt", "w") as f:
        f.write(text or "Transcription failed")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
