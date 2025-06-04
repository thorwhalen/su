"""
Speech Utils (su) - Fundamental tools for speech recognition, generation, and audio processing.

This module provides simple, high-level interfaces to the most important speech
functionalities including recognition, text-to-speech, and audio analysis.
"""

import io
import os
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

try:
    import speech_recognition as sr
    import pyttsx3
    import librosa
    import numpy as np
    from pydub import AudioSegment
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e}. Please install with: pip install su"
    )


class SpeechRecognizer:
    """Easy-to-use speech recognition interface."""

    def __init__(self, engine: str = 'google'):
        """
        Initialize speech recognizer.

        Args:
            engine: Recognition engine ('google', 'sphinx', 'wit', 'azure', 'houndify')
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = engine

    def listen_and_recognize(self, timeout: int = 5) -> Optional[str]:
        """
        Listen to microphone and return recognized text.

        Args:
            timeout: Maximum seconds to wait for speech

        Returns:
            Recognized text or None if recognition failed

        Example:
            >>> recognizer = SpeechRecognizer()
            >>> text = recognizer.listen_and_recognize()
            >>> print(f"You said: {text}")
        """
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=timeout)

            if self.engine == 'google':
                return self.recognizer.recognize_google(audio)
            elif self.engine == 'sphinx':
                return self.recognizer.recognize_sphinx(audio)
            # Add other engines as needed

        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            return None

    def recognize_file(self, audio_file: Union[str, Path]) -> Optional[str]:
        """
        Recognize speech from audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Recognized text or None if recognition failed

        Example:
            >>> recognizer = SpeechRecognizer()
            >>> text = recognizer.recognize_file("speech.wav")
            >>> print(f"Audio contains: {text}")
        """
        try:
            with sr.AudioFile(str(audio_file)) as source:
                audio = self.recognizer.record(source)

            if self.engine == 'google':
                return self.recognizer.recognize_google(audio)
            elif self.engine == 'sphinx':
                return self.recognizer.recognize_sphinx(audio)

        except (sr.UnknownValueError, sr.RequestError, FileNotFoundError):
            return None


class TextToSpeech:
    """Easy-to-use text-to-speech interface."""

    def __init__(self, rate: int = 200, volume: float = 0.9):
        """
        Initialize text-to-speech engine.

        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def speak(self, text: str, save_to: Optional[Union[str, Path]] = None):
        """
        Convert text to speech.

        Args:
            text: Text to convert to speech
            save_to: Optional file path to save audio

        Example:
            >>> tts = TextToSpeech()
            >>> tts.speak("Hello, world!")
            >>> tts.speak("Save this", save_to="output.wav")
        """
        if save_to:
            self.engine.save_to_file(text, str(save_to))
            self.engine.runAndWait()
        else:
            self.engine.say(text)
            self.engine.runAndWait()

    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices."""
        voices = self.engine.getProperty('voices')
        return [
            {'id': v.id, 'name': v.name, 'lang': getattr(v, 'languages', [])}
            for v in voices
        ]

    def set_voice(self, voice_id: str):
        """Set voice by ID."""
        self.engine.setProperty('voice', voice_id)


class AudioProcessor:
    """Audio file processing utilities."""

    @staticmethod
    def load_audio(
        file_path: Union[str, Path], sample_rate: Optional[int] = None
    ) -> tuple:
        """
        Load audio file.

        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate (None for original)

        Returns:
            Tuple of (audio_data, sample_rate)

        Example:
            >>> audio, sr = AudioProcessor.load_audio("speech.wav")
            >>> print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
        """
        audio, sr = librosa.load(str(file_path), sr=sample_rate)
        return audio, sr

    @staticmethod
    def convert_format(
        input_path: Union[str, Path], output_path: Union[str, Path], format: str = "wav"
    ) -> bool:
        """
        Convert audio file format.

        Args:
            input_path: Input file path
            output_path: Output file path
            format: Target format (wav, mp3, flac, etc.)

        Returns:
            True if conversion successful

        Example:
            >>> AudioProcessor.convert_format("input.mp3", "output.wav")
            True
        """
        try:
            audio = AudioSegment.from_file(str(input_path))
            audio.export(str(output_path), format=format)
            return True
        except Exception:
            return False

    @staticmethod
    def extract_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract basic audio features for ML.

        Args:
            audio_data: Audio time series
            sample_rate: Sample rate

        Returns:
            Dictionary with extracted features

        Example:
            >>> audio, sr = AudioProcessor.load_audio("speech.wav")
            >>> features = AudioProcessor.extract_features(audio, sr)
            >>> print(f"MFCC shape: {features['mfcc'].shape}")
        """
        features = {}

        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        )
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate
        )
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_data)

        # Tempo and beat
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features['tempo'] = tempo
        features['beats'] = beats

        return features


# Convenience functions for quick usage
def quick_recognize(timeout: int = 5) -> Optional[str]:
    """
    Quick speech recognition from microphone.

    Example:
        >>> text = quick_recognize()
        >>> print(f"You said: {text}")
    """
    recognizer = SpeechRecognizer()
    return recognizer.listen_and_recognize(timeout)


def quick_speak(text: str):
    """
    Quick text-to-speech.

    Example:
        >>> quick_speak("Hello, world!")
    """
    tts = TextToSpeech()
    tts.speak(text)


def quick_transcribe(audio_file: Union[str, Path]) -> Optional[str]:
    """
    Quick transcription of audio file.

    Example:
        >>> text = quick_transcribe("recording.wav")
        >>> print(f"Transcription: {text}")
    """
    recognizer = SpeechRecognizer()
    return recognizer.recognize_file(audio_file)


# Export main classes and functions
__all__ = [
    'SpeechRecognizer',
    'TextToSpeech',
    'AudioProcessor',
    'quick_recognize',
    'quick_speak',
    'quick_transcribe',
]
