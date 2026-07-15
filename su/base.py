"""Base speech functionalities for su: speech recognition and text-to-speech.

This module keeps the *strict, explicit* speech-processing cores small and pairs
each with a *flexible* public wrapper that casts its inputs through
:mod:`su.casting` (Postel's principle: liberal inputs, strict cores).

The speech features here require the optional ``speech`` extra
(``pip install "su[speech]"`` — brings ``SpeechRecognition``, ``pyttsx3`` and
``pyaudio``) and the ``extract_features`` analysis needs the ``ml`` extra
(``librosa``). All such imports are **lazy**, so ``import su`` and the
audio-casting / cover-art tools work even when those extras are absent; a precise
:class:`~su._deps.MissingDependencyError` is raised only when a feature that
needs them is actually invoked.
"""

import os
from pathlib import Path
from typing import Any, BinaryIO
from collections.abc import Iterator, Callable

import numpy as np

from su._deps import require
from su.casting import to_wav_source, to_waveform, to_text, export_audio


class SpeechRecognizer:
    """Easy-to-use speech recognition interface (needs the ``speech`` extra)."""

    def __init__(self, engine: str = "google"):
        """Initialize speech recognizer.

        Args:
            engine: Recognition engine ('google', 'sphinx', 'wit', 'azure',
                'houndify').
        """
        self._sr = require(
            "speech_recognition", extra="speech", feature="speech recognition"
        )
        self.recognizer = self._sr.Recognizer()
        self._microphone = None  # created lazily (needs pyaudio / PortAudio)
        self.engine = engine

    @property
    def microphone(self):
        """The microphone source, created lazily (needs ``pyaudio``)."""
        if self._microphone is None:
            require("pyaudio", extra="speech", feature="microphone input")
            self._microphone = self._sr.Microphone()
        return self._microphone

    def _recognize(self, audio) -> str | None:
        """Run recognized-audio through the configured engine."""
        if self.engine == "google":
            return self.recognizer.recognize_google(audio)
        elif self.engine == "sphinx":
            return self.recognizer.recognize_sphinx(audio)
        # Extend here for other engines as needed.
        return self.recognizer.recognize_google(audio)

    def listen_and_recognize(self, timeout: int = 5) -> str | None:
        """Listen to the microphone and return recognized text.

        Args:
            timeout: Maximum seconds to wait for speech.

        Returns:
            Recognized text or None if recognition failed.

        Example:
            >>> recognizer = SpeechRecognizer()  # doctest: +SKIP
            >>> text = recognizer.listen_and_recognize()  # doctest: +SKIP
        """
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=timeout)
            return self._recognize(audio)
        except (
            self._sr.UnknownValueError,
            self._sr.RequestError,
            self._sr.WaitTimeoutError,
        ):
            return None

    def recognize_file(self, audio_file: str | Path | bytes | BinaryIO) -> str | None:
        """Recognize speech from an audio source of any supported form.

        Args:
            audio_file: A file path, bytes, or file-like object. Non-WAV formats
                are transcoded to WAV first (needs ffmpeg).

        Returns:
            Recognized text or None if recognition failed.

        Example:
            >>> recognizer = SpeechRecognizer()  # doctest: +SKIP
            >>> text = recognizer.recognize_file("speech.wav")  # doctest: +SKIP
        """
        try:
            with self._sr.AudioFile(to_wav_source(audio_file)) as source:
                audio = self.recognizer.record(source)
            return self._recognize(audio)
        except (self._sr.UnknownValueError, self._sr.RequestError, FileNotFoundError):
            return None


class TextToSpeech:
    """Easy-to-use text-to-speech interface (needs the ``speech`` extra)."""

    def __init__(self, rate: int = 200, volume: float = 0.9):
        """Initialize text-to-speech engine.

        Args:
            rate: Speech rate (words per minute).
            volume: Volume level (0.0 to 1.0).
        """
        self._pyttsx3 = require("pyttsx3", extra="speech", feature="text-to-speech")
        self.engine = self._pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self.rate = rate
        self.volume = volume

        # Fix for a macOS pyttsx3 bug (missing _current_text attribute).
        self._fix_macos_bug()

    def _fix_macos_bug(self):
        """Fix the macOS pyttsx3 ``_current_text`` attribute bug."""
        try:
            driver = None
            if hasattr(self.engine, "_driver"):
                driver = self.engine._driver
            elif hasattr(self.engine, "proxy") and hasattr(
                self.engine.proxy, "_driver"
            ):
                driver = self.engine.proxy._driver

            if driver and not hasattr(driver, "_current_text"):
                driver._current_text = ""
        except (AttributeError, Exception):
            # If we can't fix it, we'll handle it during speak().
            pass

    def speak(self, text: str, save_to: str | Path | None = None):
        """Convert text to speech.

        Args:
            text: Text to convert to speech.
            save_to: Optional file path to save audio.

        Example:
            >>> tts = TextToSpeech()  # doctest: +SKIP
            >>> tts.speak("Hello, world!")  # doctest: +SKIP
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if save_to:
                    self.engine.save_to_file(text, str(save_to))
                    self.engine.runAndWait()
                else:
                    self.engine.say(text)
                    self.engine.runAndWait()
                return  # Success.
            except AttributeError as e:
                if "_current_text" in str(e) and attempt < max_retries - 1:
                    # Handle the macOS bug by recreating the engine.
                    try:
                        self.engine.stop()
                    except Exception:
                        pass
                    self.engine = self._pyttsx3.init()
                    self.engine.setProperty("rate", self.rate)
                    self.engine.setProperty("volume", self.volume)
                    self._fix_macos_bug()
                    # Retry in next iteration.
                else:
                    raise e

    def get_voices(self) -> list[dict[str, Any]]:
        """Get available voices."""
        voices = self.engine.getProperty("voices")
        return [
            {"id": v.id, "name": v.name, "lang": getattr(v, "languages", [])}
            for v in voices
        ]

    def set_voice(self, voice_id: str):
        """Set voice by ID."""
        self.engine.setProperty("voice", voice_id)


class AudioProcessor:
    """Audio file processing utilities.

    ``load_audio`` and ``convert_format`` use the lightweight core (pydub);
    ``extract_features`` needs the ``ml`` extra (librosa).
    """

    @staticmethod
    def load_audio(
        file_path: str | Path | bytes | BinaryIO,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load audio to a ``(waveform, sample_rate)`` pair.

        Accepts any source :func:`su.casting.to_waveform` accepts. Uses the
        lightweight core (no librosa required).

        Args:
            file_path: Audio source (path, bytes, file-like, waveform, ...).
            sample_rate: Target sample rate (None keeps the native rate).

        Returns:
            Tuple of (audio_data, sample_rate), audio as float32 mono in [-1, 1].

        Example:
            >>> import numpy as np
            >>> from su.casting import to_wav_bytes
            >>> wav = to_wav_bytes(np.zeros(8000, dtype=np.int16), sample_rate=8000)
            >>> audio, sr = AudioProcessor.load_audio(wav)
            >>> audio.dtype, sr
            (dtype('float32'), 8000)
        """
        return to_waveform(file_path, sample_rate=sample_rate, mono=True)

    @staticmethod
    def convert_format(
        input_path: str | Path, output_path: str | Path, format: str = "wav"
    ) -> bool:
        """Convert an audio file from one format to another.

        Args:
            input_path: Input file path (or any supported audio source).
            output_path: Output file path.
            format: Target format (wav, mp3, flac, ...). Non-WAV needs ffmpeg.

        Returns:
            True if conversion succeeded, else False.

        Example:
            >>> AudioProcessor.convert_format("input.mp3", "output.wav")  # doctest: +SKIP
            True
        """
        try:
            export_audio(input_path, output_path, format=format)
            return True
        except Exception:
            return False

    @staticmethod
    def extract_features(audio_data: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """Extract basic audio features for ML (needs the ``ml`` extra).

        Args:
            audio_data: Audio time series.
            sample_rate: Sample rate.

        Returns:
            Dictionary with MFCC, spectral, tempo and beat features.

        Example:
            >>> audio, sr = AudioProcessor.load_audio("speech.wav")  # doctest: +SKIP
            >>> features = AudioProcessor.extract_features(audio, sr)  # doctest: +SKIP
        """
        librosa = require("librosa", extra="ml", feature="audio feature extraction")
        features: dict[str, Any] = {}
        features["mfcc"] = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        )
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate
        )
        features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(audio_data)
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features["tempo"] = tempo
        features["beats"] = beats
        return features


def _synthesize_wav_bytes(text: str, rate: int, volume: float) -> bytes:
    """Synthesize ``text`` to WAV bytes (mono, 16 kHz) via pyttsx3 + pydub."""
    import tempfile

    from su.casting import to_wav_bytes

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        TextToSpeech(rate=rate, volume=volume).speak(text, save_to=tmp_path)
        # Re-encode to a clean mono 16 kHz WAV for downstream compatibility.
        from pydub import AudioSegment

        seg = AudioSegment.from_file(tmp_path).set_channels(1).set_frame_rate(16000)
        return to_wav_bytes(seg)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# Convenience functions for quick usage
def recognize(timeout: int = 5, *, engine: str = "google") -> str | None:
    """Quick speech recognition from the microphone (needs the ``speech`` extra).

    Args:
        timeout: Maximum seconds to wait for speech.
        engine: Recognition engine ('google', 'sphinx', 'wit', 'azure', 'houndify').

    Example:
        >>> text = recognize()  # doctest: +SKIP
    """
    return SpeechRecognizer(engine=engine).listen_and_recognize(timeout)


def speak(
    text_src: str | Path | BinaryIO | Iterator[str],
    *,
    rate: int = 200,
    volume: float = 0.9,
    egress: Callable[[bytes], Any] | str | os.PathLike | None = None,
    send_to_speakers: bool = True,
    output_format: str | None = None,
):
    """Quick text-to-speech with flexible input and output options.

    Args:
        text_src: Any source accepted by :func:`su.casting.to_text` (direct text,
            a text-file path/Path, a file-like object, or an iterable of chunks).
        rate: Speech rate (words per minute).
        volume: Volume level (0.0 to 1.0).
        egress: Optional output handler:
            - None: no file/bytes output (just speakers, if enabled).
            - str/Path: file path to save audio; the extension (or
              ``output_format``) selects the encoding.
            - Callable: receives the synthesized WAV bytes and its return value
              is returned.
        send_to_speakers: Whether to also play the audio through the speakers.
        output_format: Explicit output format for a path ``egress`` (overrides the
            extension). Where format does not matter, use a callable ``egress`` to
            pipe the raw bytes.

    Returns:
        The egress callable's result, the written :class:`~pathlib.Path`, or None.

    Example:
        >>> speak("Hello, world!")  # doctest: +SKIP
        >>> speak("Save this", egress="out.mp3", send_to_speakers=False)  # doctest: +SKIP
        >>> audio = speak("Test", egress=lambda b: b, send_to_speakers=False)  # doctest: +SKIP
    """
    text = to_text(text_src)

    egress_result = None
    if egress is not None:
        audio_bytes = _synthesize_wav_bytes(text, rate, volume)
        if isinstance(egress, (str, os.PathLike)):
            egress_result = export_audio(audio_bytes, egress, format=output_format)
        else:
            egress_result = egress(audio_bytes)

    if send_to_speakers:
        TextToSpeech(rate=rate, volume=volume).speak(text)

    return egress_result


def transcribe(
    audio_src: str | Path | bytes | BinaryIO | Iterator[bytes] | dict[str, Any],
    *,
    engine: str = "google",
    debug: bool = False,
) -> str | None:
    """Quick transcription of audio from any supported source.

    Non-WAV inputs are transcoded to WAV via :func:`su.casting.to_wav_source`
    (needs ffmpeg) before being handed to the strict recognition core.

    Args:
        audio_src: Any source accepted by :func:`su.casting.to_audio_segment`,
            or a ``{'type': 'microphone', 'timeout': N}`` dict for live capture.
        engine: Recognition engine ('google', 'sphinx', 'wit', 'azure', 'houndify').
        debug: Print debug information and re-raise unexpected errors if True.

    Returns:
        Transcribed text, or None if transcription failed.

    Example:
        >>> text = transcribe("recording.wav")  # doctest: +SKIP
        >>> text = transcribe({'type': 'microphone', 'timeout': 5})  # doctest: +SKIP
    """
    # Special-case: live microphone capture.
    if isinstance(audio_src, dict):
        if audio_src.get("type") == "microphone":
            return recognize(timeout=audio_src.get("timeout", 5), engine=engine)
        raise ValueError(f"Unsupported audio_src dict type: {audio_src.get('type')}")

    sr = require("speech_recognition", extra="speech", feature="transcription")
    try:
        if debug:
            print(f"Transcribing with engine={engine}, source type={type(audio_src)}")
        wav_source = to_wav_source(audio_src)  # strict: WAV file-like
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_source) as source:
            audio = recognizer.record(source)
        if engine == "google":
            result = recognizer.recognize_google(audio)
        elif engine == "sphinx":
            result = recognizer.recognize_sphinx(audio)
        else:
            result = recognizer.recognize_google(audio)
        if debug:
            print(f"Recognition result: {result!r}")
        return result
    except Exception as e:
        if debug:
            import traceback

            print(f"Transcription error: {e} ({type(e).__name__})")
            traceback.print_exc()
            if not isinstance(
                e,
                (
                    sr.UnknownValueError,
                    sr.RequestError,
                    FileNotFoundError,
                    ValueError,
                    TypeError,
                ),
            ):
                raise
        return None


# Export main classes and functions
__all__ = [
    "SpeechRecognizer",
    "TextToSpeech",
    "AudioProcessor",
    "recognize",
    "speak",
    "transcribe",
]
