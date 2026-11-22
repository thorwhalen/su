"""su.words: the pipeline with each step injected (no models needed)."""

import numpy as np
import pytest

from su.words import (
    SAMPLE_RATE,
    Recognised,
    Segment,
    Word,
    is_hallucination,
    spoken_form,
    transcribe_words,
    uncovered,
)


def _decode(audio):
    return np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE


def _recognise(wave, sr):
    seg = Segment(
        "I'm signing B now",
        0.5,
        1.6,
        (
            Word("I'm", 0.5, 0.7, 0.5, 0.7),
            Word("signing", 0.7, 1.1, 0.7, 1.1),
            Word("B", 1.1, 1.3, 1.1, 1.3),
            Word("now", 1.3, 1.6, 1.3, 1.6),
        ),
    )
    bogus = Segment("Subtitles by Amara", 2.0, 2.5)
    return Recognised((seg,), speech_regions=((0.4, 1.7), (2.0, 2.9)), dropped=(bogus,))


def test_pipeline_runs_every_seam_in_order():
    calls = []

    def separate(wave, sr):
        calls.append("separate")
        return wave * 0.5, 44_100  # a separator at another rate gets resampled

    def recognise(wave, sr):
        calls.append(("recognise", sr))
        return _recognise(wave, sr)

    def align(wave, sr, segments):
        calls.append("align")
        return [
            Segment(
                s.text,
                s.start,
                s.end,
                tuple(
                    Word(
                        w.text,
                        w.start + 0.01,
                        w.end,
                        w.whisper_start,
                        w.whisper_end,
                        aligned=True,
                    )
                    for w in s.words
                ),
            )
            for s in segments
        ]

    t = transcribe_words(
        "x.wav", decode=_decode, separate=separate, recognise=recognise, align=align
    )
    assert calls == ["separate", ("recognise", SAMPLE_RATE), "align"]
    assert t.separated and t.aligned
    assert [w.text for w in t.words] == ["I'm", "signing", "B", "now"]
    assert t.words[0].start == pytest.approx(0.51) and t.words[0].whisper_start == 0.5
    assert t.text == "I'm signing B now"
    assert [s.text for s in t.dropped] == ["Subtitles by Amara"]
    # The second VAD region has no word: it is untranscribed speech.
    assert t.untranscribed == ((2.0, 2.9),)


def test_separation_and_alignment_can_be_skipped():
    t = transcribe_words(
        "x.wav", decode=_decode, recognise=_recognise, separation=False, alignment=False
    )
    assert not t.separated and not t.aligned
    assert t.words[2].start == t.words[2].whisper_start == 1.1


def test_hallucinations():
    for text in (
        "Subtitles by Amara",
        " Thanks for watching!",
        "さささ",
        "a a a a",
        "!!!",
        "",
    ):
        assert is_hallucination(text), text
    for text in ("I'm signing B now", "aaa is a battery size", "no no"):
        assert not is_hallucination(text), text


def test_spoken_letters_align_by_their_names():
    assert spoken_form("B") == "bee"
    assert spoken_form("W.") == "doubleyou"
    assert spoken_form("Okay,") == "okay"
    assert spoken_form("42") == ""


def test_uncovered_regions():
    words = [Word("a", 1.0, 1.2, 1.0, 1.2), Word("b", 1.25, 2.0, 1.25, 2.0)]
    assert uncovered([(0.9, 2.1)], words, min_len=0.3) == ()
    assert uncovered([(0.0, 4.0)], words, min_len=0.3) == ((0.0, 1.0), (2.0, 4.0))
