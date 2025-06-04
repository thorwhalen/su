"""
Speech Utils (su) - Fundamental tools for speech recognition, generation, and audio processing.

This module provides simple, high-level interfaces to the most important speech
functionalities including recognition, text-to-speech, and audio analysis.
"""

from su.base import (
    SpeechRecognizer,
    TextToSpeech,
    AudioProcessor,
    recognize,
    speak,
    transcribe,
)

# Export main classes and functions
__all__ = [
    'SpeechRecognizer',
    'TextToSpeech',
    'AudioProcessor',
    'recognize',
    'speak',
    'transcribe',
]
