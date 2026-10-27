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

- **train**: Common Voice 27.0 ja clips of split v1 (no clip, speaker or normalized sentence shared
  with val or test, and no sentence with a character 3-gram Jaccard of 0.5 or more to one of them),
  transcribed with Whisper.
- **val**: the official dev core minus sentences near a test sentence (same rule). Used for every
  choice: checkpoints, thresholds, mixing, seeds.
- **test**: the official test core minus the 500 trial clips. Transcribed and labelled once, after
  every choice is fixed; evaluated once.
- **JWTD overlap**: val / test clips whose sentence is in JWTD v2 train, which D was trained on (94 val,
  80 test sentences; `leak_check.py`), are left out of the primary metric and reported separately.
- **external holdout**: a public Japanese speech corpus other than Common Voice, transcribed with a
  recognizer other than Whisper, with its script as truth. Never used for a choice. If none can be
  used it is reported as not done. Changed on 2026-10-18 from a script read into AquaVoice by the
  user, which would take 40 minutes of their time; AquaVoice itself is therefore not tested.
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
  Amended 2026-10-19, before test was opened: C's thresholds must also keep within the budget on the
  60 error-free Kotobae texts, the condition D's shipped thresholds were chosen under. Chosen on val
  alone, C's detector threshold fell to 0.67, where it marked 3.5 per 1,000 characters of those texts
  (mostly Latin, digits and written phrasing, which Common Voice lacks), so the practical axis would
  have failed on a threshold no one would ship. The 60 texts were looked at once for this; with the
  amendment their budget holds for C and D by construction, and generalization to written text rests
  on the external holdout.
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

Done on 2026-10-18 by `leak_check.py` (reports/leak-v1.json): every cross-split count is 0 and no test
clip is transcribed. Training code reads `trainval.tsv` only; `manifest.tsv` and `test.tsv` have no read
permission until step 9. Training rows are built from train paths of `trainval.tsv`, and the builder
asserts it, so a derived sample cannot come from val or test. The pre-training data of ModernBERT-Ja and
Whisper is not public and cannot be checked; Common Voice sentences include public-domain literature
that is likely in such web corpora. C and D start from the same pre-trained model and read the same
Whisper transcripts, so this cannot favour one over the other, but absolute recall may read higher than
on unseen text.

## Amendment of 2026-10-27, before test is opened

Written before `manifest.tsv` and `test.tsv` are unsealed. Every choice below was made on train and val.

### What C is now

Fine-tuning the detector alone (the C of 2026-10-19) did not reach the goal of kotobae#98, so the work
went on to the whole check: the candidates, the scorer and how they combine. C is now the check shipped
in Kotobae (kotobae PR #121), fixed as follows:

- detector `umiyosh/modernbert-ja-130m-typo-detector` rev 33115ea, scorer
  `umiyosh/modernbert-ja-130m-homophone-scorer` rev 0abe0ed, both run through Core ML by Kotobae's own
  evaluation (`typo/eval_tests.rs`)
- candidates scored for words the detector puts at 0.05 or more; a candidate scoring more than −3
  against the written word is put into the sentence and checked again by the detector (weight 1); marks only on spans with kana or kanji
- thresholds: detector 0.995, score 7.0. Chosen by splitting val by speaker into two halves and
  letting each half choose the point for the other, under 0.85 times D's false marks on val clean clips,
  at most 1 per 1,000 characters on the 60 Kotobae texts, at most 192 marks on the JWTD correct
  sentences, and no typo subset below D; both halves chose this point

A, B and D are unchanged. One reference system is added:

- **F**: the zero-shot scoring of Fujii and Shinnou (2021): `tohoku-nlp/bert-base-japanese-whole-word-masking`, untrained for
  this task, replaces the word with one [MASK] and compares log probabilities, on C's candidates and detector. On val no point
  keeps every limit, so F runs at the point that keeps the false-mark limits alone on the dev half
  (detector 0.94, score 3.5). F is reported and takes no part in any decision. Rebuilt on 2026-10-27 on C's own Rust candidates (the val
  runs of the shipped check, screen 0.05, the zero-shot leads in place of C's), the false-mark-only point is
  the same (0.94 / 3.5, 40.8% of the dev half's homophones), so F on test rescores C's test candidates.

### The goal of kotobae#98

Added as its own judgment; the primary comparison above (C − D in span recall) is still tested as
written.

- Metric: homophone span recall on test (homophone edits of clips that are neither quarantined nor in
  JWTD train), at C's fixed thresholds.
- **Reached**: point estimate at least 60%.
- **Shown**: the lower end of the 95% interval (paired cluster bootstrap over speakers, 10,000
  resamples, seed 0) at least 60%.
- Either only counts when false marks on test clean clips stay at most 1 per 1,000 characters (point
  estimate).

For the record, the selection numbers on val at this point: 64.5% of 304 homophones on one half and
60.7% of 262 on the other, 0.98 and 0.99 false marks per 1,000 characters; D 19.4% and 24.8%. On a val
half the 95% interval of C − D in homophone recall is about ±7 points, so on test (about twice a half)
the minimum detectable difference at 80% power is about 7 points. For the goal, the interval at about
566 homophones is about ±4 points, so "shown" needs a point estimate near 64%.

### External holdout

Changed from "a public corpus other than Common Voice" left open to:
[FLEURS](https://huggingface.co/datasets/google/fleurs) ja_jp (CC BY 4.0), all three splits (3,208
read utterances of translated Wikipedia sentences). None of it was used for a choice. It is transcribed
with ReazonSpeech k2 v2 (not Whisper; the recognizer `second_asr.py` runs), its raw transcription is the truth, and the labels follow the
same `classify.py` rules. FLEURS gives no speaker ids, so its resampling unit is the sentence id.
Sentences in JWTD train are reported apart. JVS and JSUT were set aside: their audio is licensed for
non-commercial research only, and this evaluates a component of a commercial product.

### Order of the final evaluation

1. Unseal `manifest.tsv` and `test.tsv`; transcribe test with Whisper (as val) and ReazonSpeech (for
   the quarantine rule); make labels.
2. Run Kotobae's evaluation once for C, D and A; F on C's candidates; B (Jev) within the $6 cap.
3. Report every system with the statistics above, once. Nothing is chosen after this.
