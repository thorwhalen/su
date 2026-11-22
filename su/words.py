"""Word-timestamped transcription of speech, including speech under music.

``transcribe_words(audio)`` runs four steps, each a keyword-argument seam with a
working default:

1. **decode** — any container, through :func:`su.casting.to_waveform` (ffmpeg via
   pydub; bytes and file-likes never need to touch disk).
2. **separate** — keep the voice, drop the instrument: Demucs ``htdemucs``'s vocals
   stem (:func:`demucs_separator`). Pass ``separation=False`` for speech without music.
3. **recognise** — faster-whisper ``large-v3`` (int8) with word timestamps, Silero
   VAD at threshold 0.3, **no initial prompt** (Whisper copies a prompt into silent
   stretches) and ``condition_on_previous_text=False`` (:func:`whisper_recogniser`).
   Known hallucinations ("Subtitles by Amara", runs of one character) are dropped and
   kept apart.
4. **align** — torchaudio's ``MMS_FA`` forced alignment, per segment, moves each
   word's boundaries to its 20 ms frame (:func:`mms_aligner`). A spoken single
   letter is aligned by its pronunciation ("B" as "bee"). Pass ``alignment=False`` to
   keep Whisper's times.

The result (:class:`WordTranscript`) keeps Whisper's own word times beside the aligned
ones, the VAD speech regions, the speech regions no word covers ("untranscribed
speech"), and the dropped segments, so a caller can judge what to trust.

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

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from su._deps import require
from su.casting import to_waveform

#: Whisper, the VAD and MMS_FA all work at 16 kHz.
SAMPLE_RATE = 16_000
#: VAD threshold that keeps quiet speech under an instrument (Silero's default is 0.5).
DFLT_VAD_THRESHOLD = 0.3
#: Seconds of audio kept around a segment when aligning it (words at its edges).
ALIGN_PAD_S = 0.2
#: A speech region at least this long with no word in it is reported as untranscribed.
MIN_UNTRANSCRIBED_S = 0.3
#: Silence that splits the reported speech regions. Tighter than the VAD that gates
#: Whisper (which merges across 2 s), so a pause between words is not "untranscribed".
REGION_MIN_SILENCE_MS = 200
#: Padding around each reported speech region (Silero's default, 400 ms, bridges pauses).
REGION_SPEECH_PAD_MS = 30

#: Lower-cased phrases Whisper produces from silence or music (subtitle credits and the like).
KNOWN_HALLUCINATIONS: tuple[str, ...] = (
    "subtitles by amara",
    "subtitles by the amara.org community",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "transcribed by",
    "ご視聴ありがとうございました",
)

#: How a spoken single letter sounds, for aligning it by pronunciation.
LETTER_NAMES = dict(
    zip(
        "abcdefghijklmnopqrstuvwxyz",
        (
            "ay bee see dee ee eff gee aitch eye jay kay ell em en oh pee cue ar ess "
            "tee you vee doubleyou ex why zee"
        ).split(),
    )
)


@dataclass(frozen=True)
class Word:
    """One word, in seconds.

    ``start``/``end`` are the best times: aligned when alignment ran and succeeded,
    else Whisper's. ``whisper_start``/``whisper_end`` are always Whisper's own.
    """

    text: str
    start: float
    end: float
    whisper_start: float
    whisper_end: float
    probability: float | None = None
    aligned: bool = False


@dataclass(frozen=True)
class Segment:
    """A run of words Whisper decoded together."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Recognised:
    """What a recogniser returns: kept segments, VAD speech regions, dropped segments."""

    segments: tuple[Segment, ...]
    speech_regions: tuple[tuple[float, float], ...] = ()
    dropped: tuple[Segment, ...] = ()


@dataclass(frozen=True)
class WordTranscript:
    """The transcript: segments with timed words, and what to be wary of."""

    segments: tuple[Segment, ...]
    speech_regions: tuple[tuple[float, float], ...] = ()
    #: VAD speech regions (or parts of them) that no kept word covers.
    untranscribed: tuple[tuple[float, float], ...] = ()
    #: Segments dropped as known hallucinations.
    dropped: tuple[Segment, ...] = ()
    separated: bool = False
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
Recogniser = Callable[[np.ndarray, int], Recognised]
Aligner = Callable[[np.ndarray, int, Sequence[Segment]], list[Segment]]


# ---- hallucinations ------------------------------------------------------------


def is_hallucination(text: str) -> bool:
    """Whether a segment's text is one of Whisper's known hallucinations.

    That is a credit phrase, or a run of one repeated character or token
    ("さささ", "a a a a").

    >>> is_hallucination("Subtitles by Amara")
    True
    >>> is_hallucination("さささ")
    True
    >>> is_hallucination("I'm signing B now")
    False
    """
    t = text.strip().lower()
    if not t:
        return True
    if any(p in t for p in KNOWN_HALLUCINATIONS):
        return True
    chars = re.sub(r"[\s\W_]+", "", t)
    if not chars:  # punctuation only ("...", "!!!"): Whisper's music filler
        return True
    if len(chars) >= 3 and len(set(chars)) == 1:
        return True
    tokens = t.split()
    return len(tokens) >= 4 and len(set(tokens)) == 1


# ---- the default steps ---------------------------------------------------------


def _torch_device(torch, device: str | None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def demucs_separator(
    *, model: str = "htdemucs", stem: str = "vocals", device: str | None = None
) -> Separator:
    """A separator keeping one Demucs stem (default: ``htdemucs``'s vocals).

    Runs on MPS or CUDA when available. The model is loaded on first use.
    """
    cache: dict = {}

    def separate(wave: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        feature = "source separation"
        torch = require("torch", extra="transcribe", feature=feature)
        pretrained = require("demucs.pretrained", extra="transcribe", feature=feature)
        apply = require("demucs.apply", extra="transcribe", feature=feature)
        taf = require("torchaudio.functional", extra="transcribe", feature=feature)
        if "m" not in cache:
            cache["m"] = pretrained.get_model(model)
            cache["m"].eval()
        m = cache["m"]
        x = torch.from_numpy(np.asarray(wave, dtype=np.float32))
        if sr != m.samplerate:
            x = taf.resample(x, sr, m.samplerate)
        x = x.unsqueeze(0).repeat(m.audio_channels, 1)  # mono → the model's channels
        with torch.no_grad():
            out = apply.apply_model(
                m, x[None], device=_torch_device(torch, device), progress=False
            )[0]
        voice = out[m.sources.index(stem)].mean(0)
        return voice.cpu().numpy(), m.samplerate

    return separate


def whisper_recogniser(
    *,
    model: str = "large-v3",
    compute_type: str = "int8",
    vad_threshold: float = DFLT_VAD_THRESHOLD,
    language: str | None = None,
    device: str = "auto",
) -> Recogniser:
    """A faster-whisper recogniser with word timestamps and Silero VAD.

    Tuned for speech under music: no initial prompt, no conditioning on previous
    text. The model is loaded on first use.
    """
    cache: dict = {}

    def recognise(wave: np.ndarray, sr: int) -> Recognised:
        feature = "word transcription"
        fw = require("faster_whisper", extra="transcribe", feature=feature)
        vad = require("faster_whisper.vad", extra="transcribe", feature=feature)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"whisper_recogniser wants {SAMPLE_RATE} Hz audio, got {sr}"
            )
        if "m" not in cache:
            cache["m"] = fw.WhisperModel(
                model, device=device, compute_type=compute_type
            )
        audio = np.asarray(wave, dtype=np.float32)
        opts = vad.VadOptions(
            threshold=vad_threshold,
            min_silence_duration_ms=REGION_MIN_SILENCE_MS,
            speech_pad_ms=REGION_SPEECH_PAD_MS,
        )
        regions = tuple(
            (r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
            for r in vad.get_speech_timestamps(audio, opts)
        )
        segs, _info = cache["m"].transcribe(
            audio,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"threshold": vad_threshold},
            condition_on_previous_text=False,
            initial_prompt=None,
        )
        kept, dropped = [], []
        for s in segs:
            words = tuple(
                Word(w.word.strip(), w.start, w.end, w.start, w.end, w.probability)
                for w in (s.words or [])
                if w.word.strip()
            )
            seg = Segment(s.text.strip(), s.start, s.end, words)
            (dropped if is_hallucination(s.text) else kept).append(seg)
        return Recognised(tuple(kept), regions, tuple(dropped))

    return recognise


def spoken_form(word: str) -> str:
    """The letters MMS_FA aligns for a word.

    Lower-case a–z and apostrophes; a single letter becomes its name.

    >>> spoken_form("B"), spoken_form("I'm,"), spoken_form("42")
    ('bee', "i'm", '')
    """
    letters = re.sub(r"[^a-z']", "", word.lower())
    if len(letters) == 1 and letters in LETTER_NAMES:
        return LETTER_NAMES[letters]
    return letters


def mms_aligner(*, device: str | None = None, pad: float = ALIGN_PAD_S) -> Aligner:
    """An aligner using torchaudio's ``MMS_FA`` (multilingual, romanised a–z).

    Each segment is aligned on its own audio, padded by ``pad`` seconds. A word with
    no alignable letters, or a segment whose alignment fails, keeps Whisper's times.
    The model is loaded on first use.
    """
    cache: dict = {}

    def align(wave: np.ndarray, sr: int, segments: Sequence[Segment]) -> list[Segment]:
        torch = require("torch", extra="transcribe", feature="forced alignment")
        ta = require("torchaudio", extra="transcribe", feature="forced alignment")
        bundle = ta.pipelines.MMS_FA
        if sr != bundle.sample_rate:
            raise ValueError(
                f"mms_aligner wants {bundle.sample_rate} Hz audio, got {sr}"
            )
        if "model" not in cache:
            dev = device or "cpu"
            cache.update(
                model=bundle.get_model().to(dev),
                tokenizer=bundle.get_tokenizer(),
                aligner=bundle.get_aligner(),
                dev=dev,
            )
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
            clip = torch.from_numpy(samples)[None]
            try:
                with torch.inference_mode():
                    emission, _ = cache["model"](clip.to(cache["dev"]))
                tokens = cache["tokenizer"]([s for _, s in alignable])
                spans = cache["aligner"](emission[0], tokens)
            except Exception:  # noqa: BLE001 — one bad segment keeps Whisper's times
                out.append(seg)
                continue
            frame_s = clip.shape[1] / emission.shape[1] / sr
            words = list(seg.words)
            for (i, _), word_spans in zip(alignable, spans):
                start = t0 + word_spans[0].start * frame_s
                end = t0 + word_spans[-1].end * frame_s
                words[i] = replace(words[i], start=start, end=end, aligned=True)
            out.append(replace(seg, words=tuple(words)))
        return out

    return align


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
    return to_waveform(audio, sample_rate=SAMPLE_RATE)


def transcribe_words(
    audio,
    *,
    decode: Callable[[object], tuple[np.ndarray, int]] = _decode,
    separate: Separator | None = None,
    recognise: Recogniser | None = None,
    align: Aligner | None = None,
    separation: bool = True,
    alignment: bool = True,
) -> WordTranscript:
    """Transcribe ``audio`` into words with times.

    Args:
        audio: any source :func:`su.casting.to_waveform` accepts (a path, bytes,
            a file-like, a ``(samples, sample_rate)`` pair).
        decode: audio → mono ``(samples, 16000)``.
        separate: keep the voice (default :func:`demucs_separator`).
        recognise: words and VAD regions (default :func:`whisper_recogniser`).
        align: refine word times (default :func:`mms_aligner`).
        separation: ``False`` skips separation, for speech without music.
        alignment: ``False`` keeps Whisper's word times.
    """
    wave, sr = decode(audio)
    separated = False
    if separation:
        voice, vsr = (separate or demucs_separator())(wave, sr)
        if vsr != SAMPLE_RATE:
            voice, vsr = to_waveform(
                (np.asarray(voice, dtype=np.float32), vsr), sample_rate=SAMPLE_RATE
            )
        wave, sr, separated = voice, vsr, True
    rec = (recognise or whisper_recogniser())(wave, sr)
    segments = list(rec.segments)
    aligned = False
    if alignment and segments:
        segments = (align or mms_aligner())(wave, sr, segments)
        aligned = True
    words = [w for s in segments for w in s.words]
    return WordTranscript(
        segments=tuple(segments),
        speech_regions=rec.speech_regions,
        untranscribed=uncovered(rec.speech_regions, words, min_len=MIN_UNTRANSCRIBED_S),
        dropped=rec.dropped,
        separated=separated,
        aligned=aligned,
    )
