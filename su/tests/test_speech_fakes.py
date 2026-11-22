"""su.base speech classes and helpers with fake speech_recognition / pyttsx3 / librosa modules."""

import io
import sys
import types

import numpy as np
import pytest

from su import _deps, base
from su._deps import MissingDependencyError, require


class _Unknown(Exception):
    pass


class _ReqErr(Exception):
    pass


class _Timeout(Exception):
    pass


class FakeRecognizer:
    result = "hello world"
    raises = None

    def __init__(self):
        self.calls = []

    def recognize_google(self, audio):
        self.calls.append("google")
        if FakeRecognizer.raises:
            raise FakeRecognizer.raises
        return FakeRecognizer.result

    def recognize_sphinx(self, audio):
        self.calls.append("sphinx")
        return "sphinx " + FakeRecognizer.result

    def adjust_for_ambient_noise(self, source):
        pass

    def listen(self, source, timeout=None):
        return "AUDIO"

    def record(self, source):
        return "AUDIO"


class FakeAudioFile:
    def __init__(self, src):
        self.src = src

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeMic(FakeAudioFile):
    def __init__(self):
        pass


@pytest.fixture
def sr_module(monkeypatch):
    sr = types.ModuleType("speech_recognition")
    sr.Recognizer = FakeRecognizer
    sr.AudioFile = FakeAudioFile
    sr.Microphone = FakeMic
    sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError = _Unknown, _ReqErr, _Timeout
    monkeypatch.setitem(sys.modules, "speech_recognition", sr)
    monkeypatch.setitem(sys.modules, "pyaudio", types.ModuleType("pyaudio"))
    FakeRecognizer.raises = None
    FakeRecognizer.result = "hello world"
    monkeypatch.setattr(base, "to_wav_source", lambda src: io.BytesIO(b"wav"))
    return sr


def test_require_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "speech_recognition", None)
    with pytest.raises(MissingDependencyError, match=r"su\[speech\]"):
        require("speech_recognition", extra="speech", feature="transcription")
    with pytest.raises(MissingDependencyError, match="pip install"):
        require("surely_not_a_module_xyz")
    assert require("io").__name__ == "io"


def test_recognize_file_engines_and_failures(sr_module):
    r = base.SpeechRecognizer()
    assert r.recognize_file("x.wav") == "hello world"
    assert base.SpeechRecognizer(engine="sphinx").recognize_file("x.wav") == "sphinx hello world"
    assert base.SpeechRecognizer(engine="wit").recognize_file("x.wav") == "hello world"
    FakeRecognizer.raises = _Unknown()
    assert r.recognize_file("x.wav") is None


def test_listen_and_recognize_swallows_recognition_errors(sr_module):
    r = base.SpeechRecognizer()
    assert r.listen_and_recognize() == "hello world"
    assert r.microphone is r.microphone  # created once
    for exc in (_Unknown(), _ReqErr(), _Timeout()):
        FakeRecognizer.raises = exc
        assert r.listen_and_recognize() is None


def test_transcribe_paths(sr_module, capsys):
    assert base.transcribe("x.wav") == "hello world"
    assert base.transcribe("x.wav", engine="sphinx") == "sphinx hello world"
    assert base.transcribe("x.wav", debug=True) == "hello world"
    assert "Recognition result" in capsys.readouterr().out
    FakeRecognizer.raises = _Unknown()
    assert base.transcribe("x.wav") is None
    assert base.transcribe("x.wav", debug=True) is None  # expected error type: not re-raised
    FakeRecognizer.raises = KeyError("unexpected")
    assert base.transcribe("x.wav") is None
    with pytest.raises(KeyError):
        base.transcribe("x.wav", debug=True)  # unexpected errors surface in debug mode
    with pytest.raises(ValueError, match="Unsupported audio_src"):
        base.transcribe({"type": "carrier pigeon"})


def test_transcribe_microphone_dict_and_recognize(sr_module):
    assert base.transcribe({"type": "microphone", "timeout": 1}) == "hello world"
    assert base.recognize(timeout=1, engine="sphinx") == "sphinx hello world"


class FakeEngine:
    def __init__(self):
        self.props, self.said, self.saved, self.stopped = {}, [], [], False
        self.fail_times = 0

    def setProperty(self, k, v):
        self.props[k] = v

    def getProperty(self, k):
        return [types.SimpleNamespace(id="v1", name="Voice One", languages=["en"]),
                types.SimpleNamespace(id="v2", name="Voice Two")]

    def say(self, text):
        if self.fail_times:
            self.fail_times -= 1
            raise AttributeError("'X' object has no attribute '_current_text'")
        self.said.append(text)

    def save_to_file(self, text, path):
        self.saved.append((text, path))

    def runAndWait(self):
        pass

    def stop(self):
        self.stopped = True


@pytest.fixture
def tts_module(monkeypatch):
    engines = []
    mod = types.ModuleType("pyttsx3")

    def init():
        e = FakeEngine()
        engines.append(e)
        return e

    mod.init = init
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    return engines


def test_text_to_speech_basics(tts_module):
    tts = base.TextToSpeech(rate=150, volume=0.5)
    assert tts.engine.props == {"rate": 150, "volume": 0.5}
    tts.speak("hi")
    tts.speak("to file", save_to="out.wav")
    assert tts.engine.said == ["hi"] and tts.engine.saved == [("to file", "out.wav")]
    assert tts.get_voices() == [
        {"id": "v1", "name": "Voice One", "lang": ["en"]},
        {"id": "v2", "name": "Voice Two", "lang": []},
    ]
    tts.set_voice("v2")
    assert tts.engine.props["voice"] == "v2"


def test_speak_retries_once_on_the_macos_bug(tts_module):
    tts = base.TextToSpeech()
    tts.engine.fail_times = 1
    tts.speak("retry me")
    assert len(tts_module) == 2 and tts_module[0].stopped  # engine was recreated
    assert tts_module[1].said == ["retry me"]


def test_speak_gives_up_after_the_retry(tts_module):
    tts = base.TextToSpeech()
    tts.engine.fail_times = 5

    def failing_init():
        e = FakeEngine()
        e.fail_times = 5
        tts_module.append(e)
        return e

    tts._pyttsx3.init = failing_init
    with pytest.raises(AttributeError, match="_current_text"):
        tts.speak("never works")


def test_speak_egress_options(tts_module, monkeypatch):
    monkeypatch.setattr(base, "_synthesize_wav_bytes", lambda text, rate, volume: b"WAV:" + text.encode())
    assert base.speak("hi", egress=lambda b: b, send_to_speakers=False) == b"WAV:hi"
    exported = {}
    monkeypatch.setattr(base, "export_audio", lambda b, path, format=None: exported.update(b=b, p=path, f=format) or "written")
    assert base.speak("hi", egress="out.mp3", send_to_speakers=False, output_format="mp3") == "written"
    assert exported == {"b": b"WAV:hi", "p": "out.mp3", "f": "mp3"}
    assert base.speak("hi", send_to_speakers=True) is None  # speakers only
    assert tts_module[-1].said == ["hi"]


def test_audio_processor_convert_and_features(monkeypatch):
    monkeypatch.setattr(base, "export_audio", lambda *a, **k: None)
    assert base.AudioProcessor.convert_format("a.mp3", "b.wav") is True

    def boom(*a, **k):
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(base, "export_audio", boom)
    assert base.AudioProcessor.convert_format("a.mp3", "b.wav") is False

    librosa = types.ModuleType("librosa")
    feat = types.SimpleNamespace(
        mfcc=lambda **k: "mfcc", spectral_centroid=lambda **k: "sc",
        spectral_rolloff=lambda **k: "sr", zero_crossing_rate=lambda y: "zcr",
    )
    librosa.feature = feat
    librosa.beat = types.SimpleNamespace(beat_track=lambda **k: (120.0, [1, 2]))
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    out = base.AudioProcessor.extract_features(np.zeros(10), 8000)
    assert out["tempo"] == 120.0 and out["mfcc"] == "mfcc" and out["beats"] == [1, 2]
