"""su.words: the pipeline with each step injected (no models needed)."""

import numpy as np
import pytest

from su.words import (
    SAMPLE_RATE,
    Segment,
    Word,
    is_hallucination,
    spoken_form,
    transcribe_words,
    uncovered,
)


def _decode(audio):
    return np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE


def _speech(wave, sr):
    return ((0.4, 1.7), (2.0, 2.9))


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
    return [seg, Segment("Subtitles by Amara", 2.0, 2.5)]


def test_pipeline_runs_every_seam_in_order():
    calls = []

    def separate(wave, sr):
        calls.append("separate")
        return wave * 0.5, SAMPLE_RATE

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
        "x.wav",
        decode=_decode,
        separate=separate,
        detect_speech=_speech,
        recognise=recognise,
        align=align,
    )
    assert calls == ["separate", ("recognise", SAMPLE_RATE), "align"]
    assert t.separated and t.aligned
    assert [w.text for w in t.words] == ["I'm", "signing", "B", "now"]
    assert t.words[0].start == pytest.approx(0.51) and t.words[0].whisper_start == 0.5
    assert t.text == "I'm signing B now"
    assert [s.text for s in t.dropped] == ["Subtitles by Amara"]
    # The second speech region has no kept word: it is untranscribed speech.
    assert t.untranscribed == ((2.0, 2.9),)


def test_any_decoded_rate_reaches_the_recogniser_at_16k():
    seen = []

    def decode(audio):
        return np.zeros(44_100, dtype=np.float32), 44_100

    def recognise(wave, sr):
        seen.append((sr, len(wave)))
        return []

    transcribe_words(
        "x", decode=decode, detect_speech=_speech, recognise=recognise, separation=False
    )
    assert seen == [(SAMPLE_RATE, SAMPLE_RATE)]


def test_separation_and_alignment_can_be_skipped():
    t = transcribe_words(
        "x.wav",
        decode=_decode,
        detect_speech=_speech,
        recognise=_recognise,
        separation=False,
        alignment=False,
    )
    assert not t.separated and not t.aligned
    assert t.words[2].start == t.words[2].whisper_start == 1.1


def test_alignment_flag_reflects_words_actually_aligned():
    t = transcribe_words(
        "x.wav",
        decode=_decode,
        detect_speech=_speech,
        recognise=_recognise,
        separation=False,
        align=lambda wave, sr, segs: list(segs),  # an aligner that aligned nothing
    )
    assert not t.aligned


def test_empty_audio_gives_an_empty_transcript():
    t = transcribe_words(
        "x", decode=lambda a: (np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    )
    assert t.segments == () and t.words == []


def test_custom_drop_rule():
    t = transcribe_words(
        "x.wav",
        decode=_decode,
        detect_speech=_speech,
        recognise=_recognise,
        drop=lambda text: False,
        separation=False,
        alignment=False,
    )
    assert t.dropped == () and len(t.segments) == 2


def test_hallucinations():
    for text in (
        "Subtitles by Amara.",
        " Thanks for watching!",
        "さささささささささ",
        "a " * 9,
        "!!!",
        "",
        "...",
    ):
        assert is_hallucination(text), text
    for text in (
        "I'm signing B now",
        "no no no no",
        "la la la la",
        "Mmm.",
        "哈哈哈",
        "Please subscribe to my channel, and today we cook rice.",
        "And it was transcribed by hand.",
    ):
        assert not is_hallucination(text), text


def test_spoken_letters_align_by_their_names():
    assert spoken_form("B") == "bee"
    assert spoken_form("W.") == "doubleyou"
    assert spoken_form("I") == "eye"
    assert spoken_form("a") == "a" and spoken_form("A") == "a"
    assert spoken_form("Okay,") == "okay"
    assert spoken_form("naïve") == "naive"
    assert spoken_form("42") == ""


def test_uncovered_regions():
    words = [Word("a", 1.0, 1.2, 1.0, 1.2), Word("b", 1.25, 2.0, 1.25, 2.0)]
    assert uncovered([(0.9, 2.1)], words, min_len=0.3) == ()
    assert uncovered([(0.0, 4.0)], words, min_len=0.3) == ((0.0, 1.0), (2.0, 4.0))
