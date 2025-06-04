"""Base functionalities for the su module."""

import io
import os
from typing import Optional, Union, List, Dict, Any, BinaryIO, Iterator, Callable
from pathlib import Path

try:
    import speech_recognition as sr
    import pyttsx3
    import librosa
    import numpy as np
    from pydub import AudioSegment
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e}. "
        f"Please install with `pip` (or install everyghing with `pip install su`"
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
            >>> text = recognizer.listen_and_recognize()  # doctest: +SKIP
            >>> print(f"You said: {text}")  # doctest: +SKIP
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
            >>> text = recognizer.recognize_file("speech.wav")  # doctest: +SKIP
            >>> print(f"Audio contains: {text}")  # doctest: +SKIP
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
        self.rate = rate
        self.volume = volume

        # Fix for macOS pyttsx3 bug
        self._fix_macos_bug()

    def _fix_macos_bug(self):
        """Fix the macOS pyttsx3 _current_text attribute bug."""
        try:
            # Try to access the driver through different possible paths
            driver = None
            if hasattr(self.engine, '_driver'):
                driver = self.engine._driver
            elif hasattr(self.engine, 'proxy') and hasattr(
                self.engine.proxy, '_driver'
            ):
                driver = self.engine.proxy._driver

            if driver and not hasattr(driver, '_current_text'):
                driver._current_text = ""
        except (AttributeError, Exception):
            # If we can't fix it, we'll handle it during speak()
            pass

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
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if save_to:
                    self.engine.save_to_file(text, str(save_to))
                    self.engine.runAndWait()
                else:
                    self.engine.say(text)
                    self.engine.runAndWait()
                return  # Success, exit the function

            except AttributeError as e:
                if "_current_text" in str(e) and attempt < max_retries - 1:
                    # Handle the macOS bug by recreating the engine
                    try:
                        self.engine.stop()
                    except:
                        pass

                    # Recreate engine
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', self.rate)
                    self.engine.setProperty('volume', self.volume)
                    self._fix_macos_bug()
                    # Will retry in next iteration
                else:
                    # If it's not the known bug or we've exhausted retries, re-raise
                    raise e
            except Exception as e:
                # For any other exception, just re-raise
                raise e

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
            >>> audio, sr = AudioProcessor.load_audio("speech.wav")  # doctest: +SKIP
            >>> print(f"Audio shape: {audio.shape}, Sample rate: {sr}")  # doctest: +SKIP
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
            >>> AudioProcessor.convert_format("input.mp3", "output.wav")  # doctest: +SKIP
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
            >>> audio, sr = AudioProcessor.load_audio("speech.wav")  # doctest: +SKIP
            >>> features = AudioProcessor.extract_features(audio, sr)  # doctest: +SKIP
            >>> print(f"MFCC shape: {features['mfcc'].shape}")  # doctest: +SKIP
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


def _resolve_audio_src(
    audio_src: Union[str, Path, bytes, BinaryIO, Iterator[bytes], Dict[str, Any]],
) -> Union[str, BinaryIO]:
    """
    Resolve various audio source formats into a format usable by speech_recognition.

    Args:
        audio_src: Audio source in various formats:
            - str/Path: File path
            - bytes: Raw audio file bytes
            - BinaryIO: File-like object (open file, BytesIO, etc.)
            - Iterator[bytes]: Iterator yielding audio chunks
            - Dict with 'type': Special sources like {'type': 'microphone', 'timeout': 5}

    Returns:
        Either a file path (str) or a file-like object (BinaryIO) that can be used
        with speech_recognition.AudioFile

    Raises:
        ValueError: If audio_src format is not supported
        TypeError: If audio_src type is not recognized
    """
    # Handle file paths
    if isinstance(audio_src, (str, Path)):
        return str(audio_src)

    # Handle raw bytes - convert to BytesIO
    if isinstance(audio_src, bytes):
        return io.BytesIO(audio_src)

    # Handle file-like objects (already compatible)
    if hasattr(audio_src, 'read') and hasattr(audio_src, 'seek'):
        # Ensure we're at the beginning of the stream
        if hasattr(audio_src, 'seek'):
            try:
                audio_src.seek(0)
            except (io.UnsupportedOperation, OSError):
                pass  # Some streams don't support seeking
        return audio_src

    # Handle iterators of bytes chunks
    if hasattr(audio_src, '__iter__') and not isinstance(audio_src, (str, bytes)):
        try:
            # Collect all chunks into a single bytes object
            chunks = []
            for chunk in audio_src:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
                else:
                    raise ValueError(f"Iterator must yield bytes, got {type(chunk)}")
            return io.BytesIO(b''.join(chunks))
        except Exception as e:
            raise ValueError(f"Failed to process audio iterator: {e}")

    # Handle special dictionary specifications
    if isinstance(audio_src, dict):
        if audio_src.get('type') == 'microphone':
            # Return a special marker that transcribe() will handle
            return audio_src
        else:
            raise ValueError(
                f"Unsupported audio_src dict type: {audio_src.get('type')}"
            )

    # If we get here, the type is not supported
    raise TypeError(
        f"Unsupported audio_src type: {type(audio_src)}. "
        f"Supported types: str, Path, bytes, file-like objects, "
        f"iterators of bytes, or dict with 'type' key."
    )


def _resolve_text_src(text_src: Union[str, Path, BinaryIO, Iterator[str]]) -> str:
    """
    Resolve various text source formats into actual text string.

    Args:
        text_src: Text source in various formats:
            - str: Direct text, or file path if starts with os.path.sep
            - Path: File path to text file
            - BinaryIO: File-like object containing text
            - Iterator[str]: Iterator yielding text chunks

    Returns:
        The resolved text string

    Raises:
        ValueError: If text_src format is not supported or file not found
        TypeError: If text_src type is not recognized
    """
    # Handle direct text strings vs file paths
    if isinstance(text_src, str):
        # If string starts with path separator, treat as file path
        if text_src.startswith(os.path.sep) or (
            len(text_src) > 1 and text_src[1] == ':'
        ):  # Windows drive paths
            try:
                with open(text_src, 'r', encoding='utf-8') as f:
                    return f.read()
            except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to read text file '{text_src}': {e}")
        else:
            # Direct text string
            return text_src

    # Handle Path objects
    if isinstance(text_src, Path):
        try:
            return text_src.read_text(encoding='utf-8')
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to read text file '{text_src}': {e}")

    # Handle file-like objects
    if hasattr(text_src, 'read'):
        try:
            # Ensure we're at the beginning of the stream
            if hasattr(text_src, 'seek'):
                try:
                    text_src.seek(0)
                except (io.UnsupportedOperation, OSError):
                    pass

            content = text_src.read()
            # Handle both text and binary modes
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return content
        except (UnicodeDecodeError, AttributeError) as e:
            raise ValueError(f"Failed to read from text stream: {e}")

    # Handle iterators of text chunks
    if hasattr(text_src, '__iter__') and not isinstance(text_src, (str, bytes)):
        try:
            chunks = []
            for chunk in text_src:
                if isinstance(chunk, str):
                    chunks.append(chunk)
                elif isinstance(chunk, bytes):
                    chunks.append(chunk.decode('utf-8'))
                else:
                    raise ValueError(
                        f"Iterator must yield str or bytes, got {type(chunk)}"
                    )
            return ''.join(chunks)
        except Exception as e:
            raise ValueError(f"Failed to process text iterator: {e}")

    # If we get here, the type is not supported
    raise TypeError(
        f"Unsupported text_src type: {type(text_src)}. "
        f"Supported types: str, Path, file-like objects, iterators of str/bytes."
    )


def _create_audio_bytes(text: str, rate: int, volume: float) -> bytes:
    """
    Create audio bytes from text using TTS engine.

    Args:
        text: Text to convert to speech
        rate: Speech rate (words per minute)
        volume: Volume level (0.0 to 1.0)

    Returns:
        Audio bytes in WAV format compatible with speech_recognition
    """
    # Create a temporary file to capture audio bytes
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Use TextToSpeech to save to temporary file
        tts = TextToSpeech(rate=rate, volume=volume)
        tts.speak(text, save_to=tmp_path)

        # Convert to proper WAV format using pydub (more reliable)
        try:
            from pydub import AudioSegment

            # Load and re-export as proper WAV
            audio_segment = AudioSegment.from_file(tmp_path)

            # Export as WAV with specific parameters for speech_recognition compatibility
            wav_export = audio_segment.export(
                format="wav", parameters=["-ac", "1", "-ar", "16000"]  # mono, 16kHz
            )
            audio_bytes = wav_export.read()
            wav_export.close()

            return audio_bytes

        except ImportError:
            # Fallback: just read the raw file if pydub not available
            with open(tmp_path, 'rb') as f:
                return f.read()

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# Convenience functions for quick usage
def recognize(timeout: int = 5, *, engine: str = 'google') -> Optional[str]:
    """
    Quick speech recognition from microphone.

    Args:
        timeout: Maximum seconds to wait for speech
        engine: Recognition engine ('google', 'sphinx', 'wit', 'azure', 'houndify')

    Example:
        >>> text = recognize()  # doctest: +SKIP
        >>> print(f"You said: {text}")  # doctest: +SKIP

        >>> # Use different engine
        >>> text = recognize(timeout=10, engine='sphinx')  # doctest: +SKIP

        >>> # Partial application for custom recognizer
        >>> from functools import partial
        >>> fast_recognize = partial(recognize, timeout=2, engine='google')
        >>> text = fast_recognize()  # doctest: +SKIP
    """
    recognizer = SpeechRecognizer(engine=engine)
    return recognizer.listen_and_recognize(timeout)


def speak(
    text_src: Union[str, Path, BinaryIO, Iterator[str]],
    *,
    rate: int = 200,
    volume: float = 0.9,
    egress: Optional[Union[Callable[[bytes], Any], str]] = None,
    send_to_speakers: bool = True,
):
    """
    Quick text-to-speech with flexible input and output options.

    Args:
        text_src: Text source in various formats:
            - str: Direct text, or file path if starts with os.path.sep
            - Path: File path to text file
            - BinaryIO: File-like object containing text
            - Iterator[str]: Iterator yielding text chunks
        rate: Speech rate (words per minute)
        volume: Volume level (0.0 to 1.0)
        egress: Optional output handler:
            - None: No special output handling
            - str: File path to save audio
            - Callable: Function to process audio bytes
        send_to_speakers: Whether to play audio through speakers

    Returns:
        Result of egress function if provided, otherwise None

    Example:
        >>> # Direct text (default behavior - hear sound)
        >>> speak("Hello, world!")

        >>> # Custom voice settings
        >>> speak("Slow and quiet", rate=100, volume=0.5)

        >>> # File path as input
        >>> speak("/path/to/text_file.txt")  # doctest: +SKIP

        >>> # Save to file without hearing
        >>> speak("Save this", egress="output.wav", send_to_speakers=False)

        >>> # Get audio bytes
        >>> audio_bytes = speak("Test", egress=lambda x: x, send_to_speakers=False)

        >>> # Both save and hear
        >>> speak("Hello", egress="greeting.wav", send_to_speakers=True)

        >>> # Use with file-like object
        >>> from io import StringIO  # doctest: +SKIP
        >>> text_stream = StringIO("This is from a stream")  # doctest: +SKIP
        >>> speak(text_stream)  # doctest: +SKIP

        >>> # Partial application for consistent voice
        >>> from functools import partial
        >>> robot_voice = partial(speak, rate=300, volume=1.0, send_to_speakers=False, egress=lambda x: x)
        >>> audio_bytes = robot_voice("I am a robot")  # doctest: +SKIP
    """
    # Resolve text source to actual text
    text = _resolve_text_src(text_src)

    # Handle egress processing
    if egress is not None:
        # Create audio bytes for egress processing
        audio_bytes = _create_audio_bytes(text, rate, volume)

        # Handle string egress as file path
        if isinstance(egress, str):
            with open(egress, 'wb') as f:
                f.write(audio_bytes)
            egress_result = None  # File save operation returns None
        else:
            # Call the egress function
            egress_result = egress(audio_bytes)
    else:
        egress_result = None

    # Handle speaker output
    if send_to_speakers:
        tts = TextToSpeech(rate=rate, volume=volume)
        tts.speak(text)

    return egress_result


def transcribe(
    audio_src: Union[str, Path, bytes, BinaryIO, Iterator[bytes], Dict[str, Any]],
    *,
    engine: str = 'google',
    debug: bool = False,
) -> Optional[str]:
    """
    Quick transcription of audio from various sources.

    Args:
        audio_src: Audio source in various formats:
            - str/Path: File path to audio file
            - bytes: Raw audio file bytes
            - BinaryIO: File-like object (open file, BytesIO, etc.)
            - Iterator[bytes]: Iterator yielding audio chunks
            - Dict: Special sources like {'type': 'microphone', 'timeout': 5}
        engine: Recognition engine ('google', 'sphinx', 'wit', 'azure', 'houndify')
        debug: Print debug information if True

    Returns:
        Transcribed text or None if transcription failed

    Example:
        >>> # File path
        >>> text = transcribe("recording.wav")  # doctest: +SKIP
        >>> print(f"Transcription: {text}")  # doctest: +SKIP

        >>> # Raw bytes with debug
        >>> text = transcribe(audio_bytes, debug=True)  # doctest: +SKIP

        >>> # Live microphone
        >>> text = transcribe({'type': 'microphone', 'timeout': 5})  # doctest: +SKIP
    """
    try:
        if debug:
            print(f"Transcribing with engine: {engine}")
            print(f"Audio source type: {type(audio_src)}")
            if isinstance(audio_src, bytes):
                print(f"Audio bytes length: {len(audio_src)}")

        resolved_src = _resolve_audio_src(audio_src)

        if debug:
            print(f"Resolved source type: {type(resolved_src)}")

        # Handle special microphone case
        if isinstance(resolved_src, dict) and resolved_src.get('type') == 'microphone':
            timeout = resolved_src.get('timeout', 5)
            return recognize(timeout=timeout, engine=engine)

        # Handle file path or file-like object
        recognizer = SpeechRecognizer(engine=engine)

        # Use the resolved source with AudioFile
        if debug:
            print("Creating AudioFile from resolved source...")

        with sr.AudioFile(resolved_src) as source:
            if debug:
                print(f"AudioFile created successfully")
                print(f"Sample rate: {source.SAMPLE_RATE}")
                print(f"Sample width: {source.SAMPLE_WIDTH}")

            audio = recognizer.recognizer.record(source)
            if debug:
                print("Audio recorded from source")

        if engine == 'google':
            if debug:
                print("Using Google Speech Recognition...")
            result = recognizer.recognizer.recognize_google(audio)
        elif engine == 'sphinx':
            if debug:
                print("Using CMU Sphinx...")
            result = recognizer.recognizer.recognize_sphinx(audio)
        else:
            if debug:
                print(f"Using engine: {engine}")
            # For other engines, fall back to the original method
            result = recognizer.recognize_file(resolved_src)

        if debug:
            print(f"Recognition result: '{result}'")
        return result

    except Exception as e:
        if debug:
            print(f"Transcription error: {e}")
            print(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()

        # Check for specific error types
        if isinstance(
            e,
            (
                sr.UnknownValueError,
                sr.RequestError,
                FileNotFoundError,
                ValueError,
                TypeError,
            ),
        ):
            return None
        else:
            # For debugging, re-raise unexpected errors when debug=True
            if debug:
                raise e
            return None


# Export main classes and functions
__all__ = [
    'SpeechRecognizer',
    'TextToSpeech',
    'AudioProcessor',
    'recognize',
    'speak',
    'transcribe',
]
