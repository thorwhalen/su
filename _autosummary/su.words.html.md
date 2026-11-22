# su.words

Word-timestamped transcription of speech, including speech under music.

`transcribe_words(audio)` runs these steps, each a keyword-argument seam with a
working default:

1. **decode** — any container, through [`su.casting.to_waveform()`](su.casting.html.md#su.casting.to_waveform) (ffmpeg via
   pydub; bytes and file-likes never need to touch disk), at its own sample rate.
2. **separate** — keep the voice, drop the instrument: Demucs `htdemucs`’s vocals
   stem ([`demucs_separator()`](#su.words.demucs_separator)), at the model’s rate. `separation=False` skips
   it, for speech without music.
3. **detect_speech** — where someone speaks: Silero VAD at threshold 0.3, with tight
   region edges ([`silero_speech_regions()`](#su.words.silero_speech_regions)).
4. **recognise** — faster-whisper `large-v3` (int8) with word timestamps, \*\*no
   initial prompt\*\* (Whisper copies a prompt into silent stretches) and
   `condition_on_previous_text=False` ([`whisper_recogniser()`](#su.words.whisper_recogniser)).
5. **drop** — segments that are Whisper’s known hallucinations (“Subtitles by
   Amara”, long runs of one character) are set apart ([`is_hallucination()`](#su.words.is_hallucination)).
6. **align** — torchaudio’s `MMS_FA` forced alignment, per segment, moves each
   word’s boundaries to its 20 ms frame ([`mms_aligner()`](#su.words.mms_aligner)). A spoken capital
   letter is aligned by its name (“B” as “bee”). An alignment that lands far from
   Whisper’s time is not trusted. `alignment=False` keeps Whisper’s times.

The result ([`WordTranscript`](#su.words.WordTranscript)) keeps Whisper’s own word times beside the aligned
ones, the speech regions, the speech regions no word covers (“untranscribed
speech”), and the dropped segments, so a caller can judge what to trust.

The default models are made once per process and reused across calls; pass your own
seams to control that.

Needs the `transcribe` extra (demucs, faster-whisper, torch, torchaudio) and
ffmpeg. All heavy imports are lazy, so `import su.words` costs nothing.

Known limit: a single spoken letter under a chord stays unreliable (runs vary between
reruns); use such words as anchors, and label from another signal where it matters.

### Example

```pycon
>>> t = transcribe_words("session.m4a")
>>> [(w.text, round(w.start, 2)) for w in t.words][:3]
[('I', 0.52), ("I'm", 0.6), ('signing', 0.82)]
```

### Module Attributes

| [`SAMPLE_RATE`](#su.words.SAMPLE_RATE)           | Whisper, the VAD and MMS_FA all work at 16 kHz.                                        |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`DFLT_VAD_THRESHOLD`](#su.words.DFLT_VAD_THRESHOLD)    | VAD threshold that keeps quiet speech under an instrument (Silero's default is 0.5).   |
| [`REGION_MIN_SILENCE_MS`](#su.words.REGION_MIN_SILENCE_MS) | Silence that splits the reported speech regions.                                       |
| [`REGION_SPEECH_PAD_MS`](#su.words.REGION_SPEECH_PAD_MS)  | Padding around each reported speech region (Silero's default, 400 ms, bridges pauses). |
| [`ALIGN_PAD_S`](#su.words.ALIGN_PAD_S)           | Seconds of audio kept around a segment when aligning it (words at its edges).          |
| [`MAX_ALIGN_SHIFT_S`](#su.words.MAX_ALIGN_SHIFT_S)     | An aligned word further than this (seconds) from Whisper's time keeps Whisper's.       |
| [`MMS_FRAME_SAMPLES`](#su.words.MMS_FRAME_SAMPLES)     | MMS_FA's frame stride, in samples at 16 kHz (20 ms).                                   |
| [`MIN_UNTRANSCRIBED_S`](#su.words.MIN_UNTRANSCRIBED_S)   | A speech region at least this long with no word in it is reported as untranscribed.    |
| [`MIN_CHAR_RUN`](#su.words.MIN_CHAR_RUN)          | A run of one repeated character (letters only) at least this long is a hallucination.  |
| [`MIN_TOKEN_RUN`](#su.words.MIN_TOKEN_RUN)         | A run of one repeated token at least this long is a hallucination.                     |
| [`KNOWN_HALLUCINATIONS`](#su.words.KNOWN_HALLUCINATIONS)  | Phrases Whisper produces from silence or music (subtitle credits and the like).        |
| [`LETTER_NAMES`](#su.words.LETTER_NAMES)          | How a spoken capital letter sounds, for aligning it by pronunciation.                  |

### Functions

| [`demucs_separator`](#su.words.demucs_separator)(\*[, model, stem, device])        | A separator keeping one Demucs stem (default: `htdemucs`'s vocals).              |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`is_hallucination`](#su.words.is_hallucination)(text)                             | Whether a segment's text is one of Whisper's known hallucinations.               |
| [`mms_aligner`](#su.words.mms_aligner)(\*[, device, pad, max_shift])          | An aligner using torchaudio's `MMS_FA` (multilingual, romanised a–z).            |
| [`resample`](#su.words.resample)(wave, sr, target)                         | `wave` at `target` Hz (torchaudio's band-limited resampler; float, no clipping). |
| [`silero_speech_regions`](#su.words.silero_speech_regions)(\*[, threshold, ...])        | Speech regions, in seconds, from Silero VAD (as bundled with faster-whisper).    |
| [`spoken_form`](#su.words.spoken_form)(word)                                  | The letters MMS_FA aligns for a word.                                            |
| [`transcribe_words`](#su.words.transcribe_words)(audio, \*[, decode, ...])         | Transcribe `audio` into words with times.                                        |
| [`uncovered`](#su.words.uncovered)(regions, words, \*, min_len)             | The parts of `regions` no word covers, at least `min_len` long.                  |
| [`whisper_recogniser`](#su.words.whisper_recogniser)(\*[, model, compute_type, ...]) | A faster-whisper recogniser with word timestamps, gated by Silero VAD.           |

### Classes

| [`Segment`](#su.words.Segment)(text, start, end[, words])                | A run of words Whisper decoded together.                           |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`Word`](#su.words.Word)(text, start, end, whisper_start, ...[, ...]) | One word, in seconds.                                              |
| [`WordTranscript`](#su.words.WordTranscript)(segments[, speech_regions, ...])   | The transcript: segments with timed words, and what to be wary of. |

### su.words.ALIGN_PAD_S *= 0.2*

Seconds of audio kept around a segment when aligning it (words at its edges).

### su.words.DFLT_VAD_THRESHOLD *= 0.3*

VAD threshold that keeps quiet speech under an instrument (Silero’s default is 0.5).

### su.words.KNOWN_HALLUCINATIONS *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]* *= frozenset({'please subscribe', 'subtitles by amara', 'subtitles by the amaraorg community', 'thank you', 'thank you for watching', 'thanks for watching', 'ご視聴ありがとうございました'})*

Phrases Whisper produces from silence or music (subtitle credits and the like). A
segment is dropped only when its whole text, without punctuation, is one of these.

### su.words.LETTER_NAMES *= {'b': 'bee', 'c': 'see', 'd': 'dee', 'e': 'ee', 'f': 'eff', 'g': 'gee', 'h': 'aitch', 'i': 'eye', 'j': 'jay', 'k': 'kay', 'l': 'ell', 'm': 'em', 'n': 'en', 'o': 'oh', 'p': 'pee', 'q': 'cue', 'r': 'ar', 's': 'ess', 't': 'tee', 'u': 'you', 'v': 'vee', 'w': 'doubleyou', 'x': 'ex', 'y': 'why', 'z': 'zee'}*

How a spoken capital letter sounds, for aligning it by pronunciation. “A” is left
out: at a sentence start it is far more often the article.

### su.words.MAX_ALIGN_SHIFT_S *= 1.0*

An aligned word further than this (seconds) from Whisper’s time keeps Whisper’s.

### su.words.MIN_CHAR_RUN *= 8*

A run of one repeated character (letters only) at least this long is a hallucination.

### su.words.MIN_TOKEN_RUN *= 8*

A run of one repeated token at least this long is a hallucination.

### su.words.MIN_UNTRANSCRIBED_S *= 0.3*

A speech region at least this long with no word in it is reported as untranscribed.

### su.words.MMS_FRAME_SAMPLES *= 320*

MMS_FA’s frame stride, in samples at 16 kHz (20 ms).

### su.words.REGION_MIN_SILENCE_MS *= 200*

Silence that splits the reported speech regions. Tighter than the VAD that gates
Whisper (which merges across 2 s), so a pause between words is not “untranscribed”.

### su.words.REGION_SPEECH_PAD_MS *= 30*

Padding around each reported speech region (Silero’s default, 400 ms, bridges pauses).

### su.words.SAMPLE_RATE *= 16000*

Whisper, the VAD and MMS_FA all work at 16 kHz.

### *class* su.words.Segment(text, start, end, words=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A run of words Whisper decoded together.

### *class* su.words.Word(text, start, end, whisper_start, whisper_end, probability=None, aligned=False, align_score=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One word, in seconds.

`start`/`end` are the best times: aligned when alignment ran and was trusted,
else Whisper’s. `whisper_start`/`whisper_end` are always Whisper’s own;
`align_score` is the aligner’s mean token score (0–1) when it ran.

### *class* su.words.WordTranscript(segments, speech_regions=(), untranscribed=(), dropped=(), separated=False, aligned=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The transcript: segments with timed words, and what to be wary of.

#### aligned *: [bool](https://docs.python.org/3/builtins/functions.html#bool)* *= False*

Whether at least one word got aligned times.

#### dropped *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Segment](#su.words.Segment), ...]* *= ()*

Segments dropped as known hallucinations.

#### *property* text *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The kept text, one string.

#### untranscribed *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), [float](https://docs.python.org/3/builtins/functions.html#float)], ...]* *= ()*

Speech regions (or parts of them) that no kept word covers.

#### *property* words *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[Word](#su.words.Word)]*

Every word, in order.

### su.words.demucs_separator(, model='htdemucs', stem='vocals', device=None)

A separator keeping one Demucs stem (default: `htdemucs`’s vocals).

It takes audio at any rate, separates at the model’s rate (44.1 kHz for
htdemucs) and returns the stem at 16 kHz. Runs on MPS or CUDA when available.
The model is loaded on first use.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### su.words.is_hallucination(text)

Whether a segment’s text is one of Whisper’s known hallucinations.

That is: nothing but punctuation, exactly a known credit phrase, or a long run of
one repeated character or token. Real speech that merely contains a phrase, or a
short repetition (“no no no”, “la la la la”), is kept.

```pycon
>>> is_hallucination("Subtitles by Amara.")
True
>>> is_hallucination("ささささささささささ")
True
>>> is_hallucination("Please subscribe to my channel, and today we cook rice.")
False
>>> is_hallucination("no no no no")
False
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### su.words.mms_aligner(, device=None, pad=0.2, max_shift=1.0)

An aligner using torchaudio’s `MMS_FA` (multilingual, romanised a–z).

Each segment is aligned on its own audio, padded by `pad` seconds. A word with
no alignable letters keeps Whisper’s times, and so does one whose aligned start
lands more than `max_shift` seconds from Whisper’s (a forced alignment always
finds *some* place). A segment whose alignment fails keeps Whisper’s times and
is logged. The model is loaded on first use.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](#su.words.Segment)]], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#su.words.Segment)]]

### su.words.resample(wave, sr, target)

`wave` at `target` Hz (torchaudio’s band-limited resampler; float, no clipping).

* **Return type:**
  `ndarray`

### su.words.silero_speech_regions(, threshold=0.3, min_silence_ms=200, speech_pad_ms=30)

Speech regions, in seconds, from Silero VAD (as bundled with faster-whisper).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]

### su.words.spoken_form(word)

The letters MMS_FA aligns for a word.

Accents are folded, then lower-case a–z and apostrophes are kept; a capital
letter on its own (other than “A”) becomes its name.

```pycon
>>> spoken_form("B"), spoken_form("I'm,"), spoken_form("café"), spoken_form("a")
('bee', "i'm", 'cafe', 'a')
>>> spoken_form("42")
''
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### su.words.transcribe_words(audio, \*, decode=<function \_decode>, separate=None, detect_speech=None, recognise=None, drop=<function is_hallucination>, align=None, separation=True, alignment=True)

Transcribe `audio` into words with times.

* **Parameters:**
  * **audio** – any source [`su.casting.to_waveform()`](su.casting.html.md#su.casting.to_waveform) accepts (a path, bytes,
    a file-like, a `(samples, sample_rate)` pair).
  * **decode** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`object`](https://docs.python.org/3/builtins/functions.html#object)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]]) – audio → mono `(samples, sample_rate)`, at any rate.
  * **separate** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – keep the voice, returning 16 kHz (default [`demucs_separator()`](#su.words.demucs_separator)).
  * **detect_speech** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – speech regions in seconds (default [`silero_speech_regions()`](#su.words.silero_speech_regions)).
  * **recognise** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](#su.words.Segment)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – segments with timed words (default [`whisper_recogniser()`](#su.words.whisper_recogniser)).
  * **drop** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – `text -> bool`, which segments to set apart as hallucinations.
  * **align** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](#su.words.Segment)]], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#su.words.Segment)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – refine word times (default [`mms_aligner()`](#su.words.mms_aligner)).
  * **separation** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – `False` skips separation, for speech without music.
  * **alignment** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – `False` keeps Whisper’s word times.
* **Return type:**
  [`WordTranscript`](#su.words.WordTranscript)

### su.words.uncovered(regions, words, , min_len)

The parts of `regions` no word covers, at least `min_len` long.

```pycon
>>> w = Word("hi", 1.0, 1.5, 1.0, 1.5)
>>> uncovered([(0.0, 3.0)], [w], min_len=0.3)
((0.0, 1.0), (1.5, 3.0))
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

### su.words.whisper_recogniser(, model='large-v3', compute_type='int8', vad_threshold=0.3, language=None, device='auto')

A faster-whisper recogniser with word timestamps, gated by Silero VAD.

Tuned for speech under music: no initial prompt, no conditioning on previous
text. The model is loaded on first use.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](#su.words.Segment)]]
