"""Word-timestamped transcription of speech, including speech under music.

``transcribe_words(audio)`` runs these steps, each a keyword-argument seam with a
working default:

1. **decode** — any container, through :func:`su.casting.to_waveform` (ffmpeg via
   pydub; bytes and file-likes never need to touch disk), at its own sample rate.
2. **separate** — keep the voice, drop the instrument: Demucs ``htdemucs``'s vocals
   stem (:func:`demucs_separator`), at the model's rate. ``separation=False`` skips
   it, for speech without music.
3. **detect_speech** — where someone speaks: Silero VAD at threshold 0.3, with tight
   region edges (:func:`silero_speech_regions`).
4. **recognise** — faster-whisper ``large-v3`` (int8) with word timestamps, **no
   initial prompt** (Whisper copies a prompt into silent stretches) and
   ``condition_on_previous_text=False`` (:func:`whisper_recogniser`).
5. **drop** — segments that are Whisper's known hallucinations ("Subtitles by
   Amara", long runs of one character) are set apart (:func:`is_hallucination`).
6. **align** — torchaudio's ``MMS_FA`` forced alignment, per segment, moves each
   word's boundaries to its 20 ms frame (:func:`mms_aligner`). A spoken capital
   letter is aligned by its name ("B" as "bee"). An alignment that lands far from
   Whisper's time is not trusted. ``alignment=False`` keeps Whisper's times.

The result (:class:`WordTranscript`) keeps Whisper's own word times beside the aligned
ones, the speech regions, the speech regions no word covers ("untranscribed
speech"), and the dropped segments, so a caller can judge what to trust.

The default models are made once per process and reused across calls; pass your own
seams to control that.

Needs the ``transcribe`` extra (demucs, faster-whisper, torch, torchaudio) and
ffmpeg. All heavy imports are lazy, so ``import su.words`` costs nothing.

Known limit: a single spoken letter under a chord stays unreliable (runs vary between
reruns); use such words as anchors, and label from another signal where it matters.

Example:
    >>> t = transcribe_words("session.m4a")  # doctest: +SKIP
    >>> [(w.text, round(w.start, 2)) for w in t.words][:3]  # doctest: +SKIP
    [('I', 0.52), ("I'm", 0.6), ('signing', 0.82)]
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache

import numpy as np

from su._deps import require
from su.casting import to_waveform

log = logging.getLogger(__name__)

#: Whisper, the VAD and MMS_FA all work at 16 kHz.
SAMPLE_RATE = 16_000
#: VAD threshold that keeps quiet speech under an instrument (Silero's default is 0.5).
DFLT_VAD_THRESHOLD = 0.3
#: Silence that splits the reported speech regions. Tighter than the VAD that gates
#: Whisper (which merges across 2 s), so a pause between words is not "untranscribed".
REGION_MIN_SILENCE_MS = 200
#: Padding around each reported speech region (Silero's default, 400 ms, bridges pauses).
REGION_SPEECH_PAD_MS = 30
#: Seconds of audio kept around a segment when aligning it (words at its edges).
ALIGN_PAD_S = 0.2
#: An aligned word further than this (seconds) from Whisper's time keeps Whisper's.
MAX_ALIGN_SHIFT_S = 1.0
#: MMS_FA's frame stride, in samples at 16 kHz (20 ms).
MMS_FRAME_SAMPLES = 320
#: A speech region at least this long with no word in it is reported as untranscribed.
MIN_UNTRANSCRIBED_S = 0.3
#: A run of one repeated character (letters only) at least this long is a hallucination.
MIN_CHAR_RUN = 8
#: A run of one repeated token at least this long is a hallucination.
MIN_TOKEN_RUN = 8

#: Phrases Whisper produces from silence or music (subtitle credits and the like). A
#: segment is dropped only when its whole text, without punctuation, is one of these.
KNOWN_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "subtitles by amara",
        "subtitles by the amaraorg community",
        "thanks for watching",
        "thank you for watching",
        "thank you",
        "please subscribe",
        "ご視聴ありがとうございました",
    }
)

#: How a spoken capital letter sounds, for aligning it by pronunciation. "A" is left
#: out: at a sentence start it is far more often the article.
LETTER_NAMES = dict(
    zip(
        "bcdefghijklmnopqrstuvwxyz",
        (
            "bee see dee ee eff gee aitch eye jay kay ell em en oh pee cue ar ess "
            "tee you vee doubleyou ex why zee"
        ).split(),
    )
)


@dataclass(frozen=True)
class Word:
    """One word, in seconds.

    ``start``/``end`` are the best times: aligned when alignment ran and was trusted,
    else Whisper's. ``whisper_start``/``whisper_end`` are always Whisper's own;
    ``align_score`` is the aligner's mean token score (0–1) when it ran.
    """

    text: str
    start: float
    end: float
    whisper_start: float
    whisper_end: float
    probability: float | None = None
    aligned: bool = False
    align_score: float | None = None


@dataclass(frozen=True)
class Segment:
    """A run of words Whisper decoded together."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class WordTranscript:
    """The transcript: segments with timed words, and what to be wary of."""

    segments: tuple[Segment, ...]
    speech_regions: tuple[tuple[float, float], ...] = ()
    #: Speech regions (or parts of them) that no kept word covers.
    untranscribed: tuple[tuple[float, float], ...] = ()
    #: Segments dropped as known hallucinations.
    dropped: tuple[Segment, ...] = ()
    separated: bool = False
    #: Whether at least one word got aligned times.
    aligned: bool = False

    @property
    def words(self) -> list[Word]:
        """Every word, in order."""
        return [w for s in self.segments for w in s.words]

    @property
    def text(self) -> str:
        """The kept text, one string."""
        return " ".join(s.text.strip() for s in self.segments).strip()


Separator = Callable[[np.ndarray, int], tuple[np.ndarray, int]]
SpeechDetector = Callable[[np.ndarray, int], Sequence[tuple[float, float]]]
Recogniser = Callable[[np.ndarray, int], Sequence[Segment]]
Aligner = Callable[[np.ndarray, int, Sequence[Segment]], list[Segment]]


# ---- hallucinations ------------------------------------------------------------


def is_hallucination(text: str) -> bool:
    """Whether a segment's text is one of Whisper's known hallucinations.

    That is: nothing but punctuation, exactly a known credit phrase, or a long run of
    one repeated character or token. Real speech that merely contains a phrase, or a
    short repetition ("no no no", "la la la la"), is kept.

    >>> is_hallucination("Subtitles by Amara.")
    True
    >>> is_hallucination("ささささささささささ")
    True
    >>> is_hallucination("Please subscribe to my channel, and today we cook rice.")
    False
    >>> is_hallucination("no no no no")
    False
    """
    t = text.strip().lower()
    words = re.sub(r"[^\w\s]+", "", t).split()
    if not words:  # empty, or punctuation only ("...", "!!!"): Whisper's music filler
        return True
    if " ".join(words) in KNOWN_HALLUCINATIONS:
        return True
    letters = "".join(words)
    if len(letters) >= MIN_CHAR_RUN and len(set(letters)) == 1:
        return True
    return len(words) >= MIN_TOKEN_RUN and len(set(words)) == 1


# ---- the default steps ---------------------------------------------------------


def _torch_device(torch, device: str | None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resample(wave: np.ndarray, sr: int, target: int) -> np.ndarray:
    """``wave`` at ``target`` Hz (torchaudio's band-limited resampler; float, no clipping)."""
    if sr == target or len(wave) == 0:
        return np.asarray(wave, dtype=np.float32)
    torch = require("torch", extra="transcribe", feature="resampling")
    taf = require("torchaudio.functional", extra="transcribe", feature="resampling")
    x = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    return taf.resample(x, sr, target).numpy()


def demucs_separator(
    *, model: str = "htdemucs", stem: str = "vocals", device: str | None = None
) -> Separator:
    """A separator keeping one Demucs stem (default: ``htdemucs``'s vocals).

    It takes audio at any rate, separates at the model's rate (44.1 kHz for
    htdemucs) and returns the stem at 16 kHz. Runs on MPS or CUDA when available.
    The model is loaded on first use.
    """
    state: dict = {}

    def separate(wave: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if len(wave) == 0:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        feature = "source separation"
        torch = require("torch", extra="transcribe", feature=feature)
        pretrained = require("demucs.pretrained", extra="transcribe", feature=feature)
        apply = require("demucs.apply", extra="transcribe", feature=feature)
        if "m" not in state:
            state["m"] = pretrained.get_model(model)
            state["m"].eval()
        m = state["m"]
        x = torch.from_numpy(resample(wave, sr, m.samplerate))
        x = x.unsqueeze(0).repeat(m.audio_channels, 1)  # mono → the model's channels
        with torch.no_grad():
            out = apply.apply_model(
                m, x[None], device=_torch_device(torch, device), progress=False
            )[0]
        voice = out[m.sources.index(stem)].mean(0).cpu().numpy()
        return resample(voice, m.samplerate, SAMPLE_RATE), SAMPLE_RATE

    return separate


def silero_speech_regions(
    *,
    threshold: float = DFLT_VAD_THRESHOLD,
    min_silence_ms: int = REGION_MIN_SILENCE_MS,
    speech_pad_ms: int = REGION_SPEECH_PAD_MS,
) -> SpeechDetector:
    """Speech regions, in seconds, from Silero VAD (as bundled with faster-whisper)."""

    def detect(wave: np.ndarray, sr: int) -> tuple[tuple[float, float], ...]:
        vad = require(
            "faster_whisper.vad", extra="transcribe", feature="speech detection"
        )
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"silero_speech_regions wants {SAMPLE_RATE} Hz audio, got {sr}"
            )
        opts = vad.VadOptions(
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        audio = np.asarray(wave, dtype=np.float32)
        return tuple(
            (r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
            for r in vad.get_speech_timestamps(audio, opts)
        )

    return detect


def whisper_recogniser(
    *,
    model: str = "large-v3",
    compute_type: str = "int8",
    vad_threshold: float = DFLT_VAD_THRESHOLD,
    language: str | None = None,
    device: str = "auto",
) -> Recogniser:
    """A faster-whisper recogniser with word timestamps, gated by Silero VAD.

    Tuned for speech under music: no initial prompt, no conditioning on previous
    text. The model is loaded on first use.
    """
    state: dict = {}

    def recognise(wave: np.ndarray, sr: int) -> list[Segment]:
        fw = require("faster_whisper", extra="transcribe", feature="word transcription")
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"whisper_recogniser wants {SAMPLE_RATE} Hz audio, got {sr}"
            )
        if "m" not in state:
            state["m"] = fw.WhisperModel(
                model, device=device, compute_type=compute_type
            )
        segs, _info = state["m"].transcribe(
            np.asarray(wave, dtype=np.float32),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"threshold": vad_threshold},
            condition_on_previous_text=False,
            initial_prompt=None,
        )
        return [
            Segment(
                s.text.strip(),
                s.start,
                s.end,
                tuple(
                    Word(w.word.strip(), w.start, w.end, w.start, w.end, w.probability)
                    for w in (s.words or [])
                    if w.word.strip()
                ),
            )
            for s in segs
        ]

    return recognise


def spoken_form(word: str) -> str:
    """The letters MMS_FA aligns for a word.

    Accents are folded, then lower-case a–z and apostrophes are kept; a capital
    letter on its own (other than "A") becomes its name.

    >>> spoken_form("B"), spoken_form("I'm,"), spoken_form("café"), spoken_form("a")
    ('bee', "i'm", 'cafe', 'a')
    >>> spoken_form("42")
    ''
    """
    folded = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode()
    bare = re.sub(r"[^A-Za-z']", "", folded)
    if len(bare) == 1 and bare.isupper() and bare.lower() in LETTER_NAMES:
        return LETTER_NAMES[bare.lower()]
    return bare.lower()


def mms_aligner(
    *,
    device: str | None = None,
    pad: float = ALIGN_PAD_S,
    max_shift: float = MAX_ALIGN_SHIFT_S,
) -> Aligner:
    """An aligner using torchaudio's ``MMS_FA`` (multilingual, romanised a–z).

    Each segment is aligned on its own audio, padded by ``pad`` seconds. A word with
    no alignable letters keeps Whisper's times, and so does one whose aligned start
    lands more than ``max_shift`` seconds from Whisper's (a forced alignment always
    finds *some* place). A segment whose alignment fails keeps Whisper's times and
    is logged. The model is loaded on first use.
    """
    state: dict = {}

    def align(wave: np.ndarray, sr: int, segments: Sequence[Segment]) -> list[Segment]:
        torch = require("torch", extra="transcribe", feature="forced alignment")
        ta = require("torchaudio", extra="transcribe", feature="forced alignment")
        if not hasattr(ta.functional, "forced_align"):
            raise RuntimeError(
                "This torchaudio has no functional.forced_align (removed after 2.9); "
                "pin torchaudio<2.10 or pass align= another aligner."
            )
        bundle = ta.pipelines.MMS_FA
        if sr != bundle.sample_rate:
            raise ValueError(
                f"mms_aligner wants {bundle.sample_rate} Hz audio, got {sr}"
            )
        if "model" not in state:
            dev = device or "cpu"
            state.update(
                model=bundle.get_model().to(dev),
                tokenizer=bundle.get_tokenizer(),
                aligner=bundle.get_aligner(),
                dev=dev,
            )
        frame_s = MMS_FRAME_SAMPLES / sr
        out = []
        for seg in segments:
            alignable = [
                (i, s)
                for i, s in ((i, spoken_form(w.text)) for i, w in enumerate(seg.words))
                if s
            ]
            if not alignable:
                out.append(seg)
                continue
            t0 = max(0.0, seg.start - pad)
            t1 = min(len(wave) / sr, seg.end + pad)
            samples = np.asarray(wave[int(t0 * sr) : int(t1 * sr)], dtype=np.float32)
            if len(samples) < MMS_FRAME_SAMPLES * 2:
                out.append(seg)
                continue
            try:
                with torch.inference_mode():
                    emission, _ = state["model"](
                        torch.from_numpy(samples)[None].to(state["dev"])
                    )
                tokens = state["tokenizer"]([s for _, s in alignable])
                spans = state["aligner"](emission[0], tokens)
            except Exception as e:  # noqa: BLE001 — one bad segment keeps Whisper's times
                log.warning(
                    "su.words: aligning %r failed (%s); keeping Whisper's times",
                    seg.text,
                    e,
                )
                out.append(seg)
                continue
            words = list(seg.words)
            for (i, _), word_spans in zip(alignable, spans):
                start = t0 + word_spans[0].start * frame_s
                end = t0 + word_spans[-1].end * frame_s
                score = float(np.mean([sp.score for sp in word_spans]))
                w = words[i]
                if abs(start - w.whisper_start) > max_shift:
                    words[i] = replace(w, align_score=score)
                else:
                    words[i] = replace(
                        w, start=start, end=end, aligned=True, align_score=score
                    )
            out.append(replace(seg, words=tuple(words)))
        return out

    return align


# Defaults are made once per process, so a batch of calls loads each model once.
_default_separator = cache(demucs_separator)
_default_detector = cache(silero_speech_regions)
_default_recogniser = cache(whisper_recogniser)
_default_aligner = cache(mms_aligner)


# ---- the pipeline --------------------------------------------------------------


def uncovered(
    regions: Sequence[tuple[float, float]], words: Sequence[Word], *, min_len: float
) -> tuple[tuple[float, float], ...]:
    """The parts of ``regions`` no word covers, at least ``min_len`` long.

    >>> w = Word("hi", 1.0, 1.5, 1.0, 1.5)
    >>> uncovered([(0.0, 3.0)], [w], min_len=0.3)
    ((0.0, 1.0), (1.5, 3.0))
    """
    spans = sorted((w.start, w.end) for w in words)
    out = []
    for a, b in regions:
        cur = a
        for s, e in spans:
            if e <= cur or s >= b:
                continue
            if s - cur >= min_len:
                out.append((cur, s))
            cur = max(cur, e)
        if b - cur >= min_len:
            out.append((cur, b))
    return tuple(out)


def _decode(audio) -> tuple[np.ndarray, int]:
    return to_waveform(audio)


def transcribe_words(
    audio,
    *,
    decode: Callable[[object], tuple[np.ndarray, int]] = _decode,
    separate: Separator | None = None,
    detect_speech: SpeechDetector | None = None,
    recognise: Recogniser | None = None,
    drop: Callable[[str], bool] = is_hallucination,
    align: Aligner | None = None,
    separation: bool = True,
    alignment: bool = True,
) -> WordTranscript:
    """Transcribe ``audio`` into words with times.

    Args:
        audio: any source :func:`su.casting.to_waveform` accepts (a path, bytes,
            a file-like, a ``(samples, sample_rate)`` pair).
        decode: audio → mono ``(samples, sample_rate)``, at any rate.
        separate: keep the voice, returning 16 kHz (default :func:`demucs_separator`).
        detect_speech: speech regions in seconds (default :func:`silero_speech_regions`).
        recognise: segments with timed words (default :func:`whisper_recogniser`).
        drop: ``text -> bool``, which segments to set apart as hallucinations.
        align: refine word times (default :func:`mms_aligner`).
        separation: ``False`` skips separation, for speech without music.
        alignment: ``False`` keeps Whisper's word times.
    """
    wave, sr = decode(audio)
    separated = False
    if separation and len(wave):
        wave, sr = (separate or _default_separator())(wave, sr)
        separated = True
    wave, sr = resample(wave, sr, SAMPLE_RATE), SAMPLE_RATE
    if not len(wave):
        return WordTranscript(segments=(), separated=separated)
    regions = tuple((detect_speech or _default_detector())(wave, sr))
    kept, dropped = [], []
    for seg in (recognise or _default_recogniser())(wave, sr):
        (dropped if drop(seg.text) else kept).append(seg)
    if alignment and kept:
        kept = (align or _default_aligner())(wave, sr, kept)
    words = [w for s in kept for w in s.words]
    return WordTranscript(
        segments=tuple(kept),
        speech_regions=regions,
        untranscribed=uncovered(regions, words, min_len=MIN_UNTRANSCRIBED_S),
        dropped=tuple(dropped),
        separated=separated,
        aligned=any(w.aligned for w in words),
    )
