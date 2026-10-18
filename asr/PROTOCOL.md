# Evaluation protocol

Written before any model is trained on speech-recognition errors and before test is opened. A change
to this file after that is a new commit with the reason, and any result it affects says so.

## Task

Input: Japanese text produced by speech recognition (Whisper here, AquaVoice in Kotobae). Output:
character spans that are likely wrong, each with a score. Kotobae underlines the spans and passes them
to the later proofreading.

The detector reads text only. A wrong word that fits its context (自己 / 事故 when both read well)
cannot be told from the right one without the audio, so part of the errors is out of reach by design;
results are not read as a ceiling of speech-recognition error detection.

## Systems

| | System | Role |
| --- | --- | --- |
| A | Rules: mark a word that is the wrong side of a pair in the JWTD misconversion table, or an unknown kanji / katakana run (IPADIC) | Simple baseline |
| B | Jev, the per-bunsetsu question of Kotobae (error measure, probability ≥ 0.55), one request per clip | Baseline; on test and the external holdout only, capped at $6 (approved 2026-10-18) |
| D | Kotobae today: detector `umiyosh/modernbert-ja-130m-typo-detector` rev 2bf8db3 + candidate scoring with modernbert-ja-130m, thresholds 0.4707 / 5.13 | Main baseline |
| C | D with the detector fine-tuned on speech-recognition errors (this work); thresholds refit on val | Candidate |

C-det and D-det (the detectors alone, without candidate scoring) are reported as well. The primary
comparison is C against D, because D is what C would replace.

Kotobae shows the union of Jev's marks and the detector's marks, so C ∪ B against D ∪ B is reported
as a secondary comparison: how many more errors a user would see marked. It takes no part in the
decision.

## Data

- **train**: Common Voice 27.0 ja clips of `make_split.py` (no clip, speaker or normalized sentence
  shared with val or test), transcribed with Whisper.
- **val**: the official dev core. Used for every choice: checkpoints, thresholds, mixing, seeds.
- **test**: the official test core minus the 500 trial clips. Transcribed and labelled once, after
  every choice is fixed; evaluated once.
- **external holdout**: AquaVoice transcripts of a script read by a person, with the script as truth.
  Never used for a choice. If it cannot be collected it is reported as not done.
- **Kotobae regression sets**: `set.json` dev / holdout (JWTD, voice-style and hand-made errors) and the
  60 error-free evaluator texts (28,000 characters). Their fresh split was used once on 2026-10-15 and
  supports no claim here.

## Labels

From the word diff of `classify.py` between the script and the transcript:

- An error span is the transcript characters of a homophone, mishear or extra edit.
- A drop edit (a script word missing) has no transcript characters. Drops are kept out of span
  metrics and reported separately: a drop counts as found when a mark touches a character next to the
  gap.
- Clips judged `broken`, and clips whose difference may be the reader's rather than the recognizer's
  (rules and audit in the label report of step 5), are quarantined: in no metric.
- Error-free clips (`same`, `notation`) measure false marks.

## Metrics

All at the operating point, with the unit named every time; token, span and sentence results are not
mixed.

- **Operating point**: the thresholds at which false marks on val error-free clips are at most 1 per
  1,000 characters, the budget Kotobae uses today. For D its shipped thresholds are kept and their
  val false-mark rate is reported.
- **Primary**: span recall: error spans (homophone, mishear, extra) overlapped by at least one mark,
  over all error spans.
- **Secondary**: span precision (marks overlapping an error span over all marks); span F1; AUPRC over
  per-word scores; false marks per 1,000 characters on test error-free clips and on the 60 Kotobae
  texts; exact span match; sentence-level AUROC and F1 (a clip has an error span vs. it has a mark);
  homophone-only span recall (the errors of kotobae#98); drop detection; latency p50 / p95 for 3,000
  characters; peak memory.

## Statistics

- H0: C and D have the same population span recall at the operating point. H1: they differ.
  Two-sided, α = 0.05. One primary test, so no correction for it; secondary comparisons use Holm.
- Unit of resampling: the speaker. Test has one clip per sentence, so speakers are the dependent
  groups. Spans of a clip are never resampled on their own.
- 95% CI of each metric and of C − D: paired cluster bootstrap over speakers, 10,000 resamples, seed 0.
- p-value: paired permutation test that swaps C and D within each speaker, 10,000 permutations, seed 0.
  McNemar is not used for F1 or recall over spans, which are not independent paired binary units.
- Reported with every result: the difference, its 95% CI, the p-value and the number of spans,
  clips and speakers. A non-significant difference is not read as equality; the minimum detectable
  difference at 80% power is estimated from val before test is opened.

## Decision

Three separate axes, each reported:

1. Statistical: the 95% CI of C − D in primary span recall excludes 0.
2. Practical: C − D ≥ +5 points of span recall with false marks on the 60 Kotobae texts at most 1 per
   1,000 characters (today's budget; a smaller gain is not worth another model download for users).
3. Generalization: on the external holdout, C's span recall is not below D's (point estimate) and the
   false marks stay within the budget.

**Adopt** when all three hold. **Reject** when the CI of C − D lies entirely below +5 points, or C is
significantly worse, or C breaks the false-mark budget on the Kotobae texts. **Undecided** otherwise,
including when the external holdout could not be collected.

## Leak controls

Before training (step 4), automated and stored as JSON: exact, normalized and near-duplicate sentences
across splits; speakers across splits; audio file hashes across splits; derived samples carrying their
parent's split; the test manifest is never read by training code; overlap of val / test sentences
with the detector's earlier training data (JWTD) and what is known of the pre-training data of
ModernBERT-Ja and Whisper.
