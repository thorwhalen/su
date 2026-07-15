"""End-to-end tests for su casting and cover-art embedding.

Tests that need to encode/decode non-WAV formats require ffmpeg; they are
skipped (not failed) when ffmpeg is unavailable, so CI stays green either way.
The pure casting tests (numpy <-> WAV) need no ffmpeg and always run.
"""

import io
import shutil

import numpy as np
import pytest
from PIL import Image

from su.casting import (
    to_audio_segment,
    to_wav_bytes,
    to_wav_source,
    to_waveform,
    export_audio,
    to_text,
)
from su.cover_art import embed_image, EMBEDDABLE_FORMATS


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


needs_ffmpeg = pytest.mark.skipif(
    not _have_ffmpeg(), reason="ffmpeg not installed"
)


def _sine(freq=440, sr=8000, seconds=0.25, amp=0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32), sr


def _cover_image():
    return Image.new("RGB", (32, 32), color=(200, 30, 30))


# --------------------------------------------------------------------------
# Casting (no ffmpeg needed: everything is WAV / raw PCM)
# --------------------------------------------------------------------------
def test_int16_waveform_roundtrip():
    samples = (np.sin(np.linspace(0, 6.28, 8000)) * 16000).astype(np.int16)
    seg = to_audio_segment(samples, sample_rate=8000)
    assert seg.frame_rate == 8000 and seg.channels == 1 and seg.sample_width == 2
    wf, sr = to_waveform(seg, dtype="int16")
    assert sr == 8000 and wf.shape == (8000,)
    assert np.array_equal(wf, samples)


def test_float_waveform_normalized():
    samples, sr = _sine()
    seg = to_audio_segment(samples, sample_rate=sr)
    wf, out_sr = to_waveform(seg, dtype="float32")
    assert out_sr == sr
    assert wf.dtype == np.float32
    assert np.max(np.abs(wf)) <= 1.0
    # Values should be close to the input (int16 quantization tolerance).
    assert np.allclose(wf, samples, atol=1e-3)


def test_wav_bytes_and_source():
    samples = np.zeros(800, dtype=np.int16)
    data = to_wav_bytes(samples, sample_rate=8000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    src = to_wav_source(samples, sample_rate=8000)
    assert src.read(4) == b"RIFF"


def test_bytes_and_filelike_inputs():
    wav = to_wav_bytes(np.zeros(800, dtype=np.int16), sample_rate=8000)
    assert to_audio_segment(wav).frame_rate == 8000
    assert to_audio_segment(io.BytesIO(wav)).frame_rate == 8000


def test_sample_list_input():
    seg = to_audio_segment([0, 100, -100, 0] * 100, sample_rate=8000)
    assert seg.channels == 1 and seg.frame_rate == 8000


def test_waveform_sr_pair_input():
    samples, sr = _sine()
    seg = to_audio_segment((samples, sr))
    assert seg.frame_rate == sr


def test_unsupported_input_raises():
    with pytest.raises(TypeError):
        to_audio_segment(object())


def test_to_text_variants():
    assert to_text("hello") == "hello"
    assert to_text(io.StringIO("stream")) == "stream"
    assert to_text(iter(["a", "b"])) == "ab"


def test_export_audio_default_wav_bytes():
    data = export_audio(np.zeros(800, dtype=np.int16), sample_rate=8000)
    assert data[:4] == b"RIFF"


# --------------------------------------------------------------------------
# Cover-art embedding (needs ffmpeg to encode mp3/flac/m4a)
# --------------------------------------------------------------------------
@needs_ffmpeg
def test_object_input_falls_back_to_flac(tmp_path):
    from mutagen.flac import FLAC

    samples, sr = _sine()
    out = embed_image((samples, sr), _cover_image(), output_dir=tmp_path)
    assert out.suffix == ".flac"
    assert out.parent == tmp_path
    assert len(FLAC(str(out)).pictures) == 1


@needs_ffmpeg
def test_mp3_format_preserved_and_original_untouched(tmp_path):
    from mutagen.mp3 import MP3

    samples, sr = _sine()
    src = tmp_path / "song.mp3"
    export_audio((samples, sr), src, format="mp3")
    assert MP3(str(src)).tags is None or not MP3(str(src)).tags.getall("APIC")

    out = embed_image(src, _cover_image())
    assert out != src  # original not overwritten by default
    assert out.suffix == ".mp3"  # format preserved
    assert out.name == "song_with_image.mp3"
    assert MP3(str(out)).tags.getall("APIC")  # cover embedded
    # Original still has no cover.
    assert MP3(str(src)).tags is None or not MP3(str(src)).tags.getall("APIC")


@needs_ffmpeg
def test_overwrite_in_place(tmp_path):
    from mutagen.flac import FLAC

    samples, sr = _sine()
    src = tmp_path / "track.flac"
    export_audio((samples, sr), src, format="flac")

    out = embed_image(src, _cover_image(), overwrite=True)
    assert out == src
    assert len(FLAC(str(src)).pictures) == 1


@needs_ffmpeg
def test_explicit_output_format_m4a(tmp_path):
    from mutagen.mp4 import MP4

    samples, sr = _sine()
    out = embed_image(
        (samples, sr), _cover_image(), output_format="m4a", output_dir=tmp_path
    )
    assert out.suffix == ".m4a"
    assert MP4(str(out)).get("covr")


@needs_ffmpeg
def test_bytes_image_and_explicit_output_path(tmp_path):
    from mutagen.flac import FLAC

    buf = io.BytesIO()
    _cover_image().save(buf, format="PNG")
    samples, sr = _sine()
    dest = tmp_path / "explicit.flac"
    out = embed_image((samples, sr), buf.getvalue(), output=dest)
    assert out == dest
    assert len(FLAC(str(out)).pictures) == 1


@needs_ffmpeg
def test_overwrite_format_mismatch_raises(tmp_path):
    samples, sr = _sine()
    src = tmp_path / "song.mp3"
    export_audio((samples, sr), src, format="mp3")
    with pytest.raises(ValueError):
        embed_image(src, _cover_image(), overwrite=True, output_format="flac")


def test_explicit_output_equals_original_without_overwrite_raises(tmp_path):
    # No ffmpeg needed: the guard fires before any encoding.
    src = tmp_path / "song.mp3"
    src.write_bytes(b"ID3fakeaudio")
    with pytest.raises(ValueError):
        embed_image(src, _cover_image(), output=src)


def test_non_embeddable_output_format_raises():
    with pytest.raises(ValueError):
        embed_image(np.zeros(800, dtype=np.int16), _cover_image(), output_format="wav")


def test_embeddable_formats_frozen():
    assert EMBEDDABLE_FORMATS == frozenset(
        {"mp3", "flac", "m4a", "aiff", "ogg", "opus"}
    )
