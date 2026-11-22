"""Speech Utils (su): tools for audio I/O, speech recognition, TTS, and cover art.

The package is organised in three layers:

- :mod:`su.casting` — flexible input/output casting for audio (Postel's
  principle): normalise paths, bytes, file-like objects, numpy waveforms, sample
  lists, or ``AudioSegment`` into the strict types the base functions need, and
  control the output format when it matters.
- :mod:`su.base` — speech recognition and text-to-speech (needs the ``speech``
  extra) plus lightweight audio processing.
- :mod:`su.cover_art` — embed cover images into audio files.

The lightweight core (audio conversion + cover art) works out of the box.
Heavier capabilities live behind extras: ``pip install "su[speech]"`` (speech
recognition + TTS) and ``pip install "su[ml]"`` (librosa feature extraction).
Those imports are lazy, so ``import su`` never fails when an extra is absent.

Example:
    >>> import su
    >>> import numpy as np
    >>> seg = su.to_audio_segment(np.zeros(8000, dtype=np.int16), sample_rate=8000)
    >>> su.to_wav_bytes(seg)[:4]
    b'RIFF'
"""

from su.base import (
    SpeechRecognizer,
    TextToSpeech,
    AudioProcessor,
    recognize,
    speak,
    transcribe,
)
from su.casting import (
    to_audio_segment,
    to_wav_source,
    to_wav_bytes,
    to_waveform,
    export_audio,
    to_text,
)
from su.cover_art import embed_image, EMBEDDABLE_FORMATS
from su.words import transcribe_words, WordTranscript

__all__ = [
    # Speech (needs the `speech` extra at call time)
    "SpeechRecognizer",
    "TextToSpeech",
    "AudioProcessor",
    "recognize",
    "speak",
    "transcribe",
    # Word-timed transcription, also under music (needs the `transcribe` extra)
    "transcribe_words",
    "WordTranscript",
    # Casting (flexible I/O)
    "to_audio_segment",
    "to_wav_source",
    "to_wav_bytes",
    "to_waveform",
    "export_audio",
    "to_text",
    # Cover art
    "embed_image",
    "EMBEDDABLE_FORMATS",
]
