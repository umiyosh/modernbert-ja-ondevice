# Speech-recognition errors for the typo detector

Work in progress. The goal is to find out, with an evaluation free of leaks, whether training the
typo detector on real speech-recognition errors makes it catch errors such as 誤変換 → 5変換 that it
misses today ([kotobae#98](https://github.com/umiyosh/kotobae/issues/98)). The answer is reported as
adopt, reject or undecided under [PROTOCOL.md](PROTOCOL.md), fixed before training; [PROGRESS.md](PROGRESS.md)
records each step.

## Design constraint

The detector reads text only. An error that is a plausible word in its context (自己 / 事故 when both
fit) cannot be told from the right word without the audio, so some errors are out of its reach by
design.

## Data

- [Common Voice Scripted Speech 27.0 Japanese](https://mozilladatacollective.com/datasets/cmu5xbhnf00fmnq07z9p3aa70)
  (CC0-1.0). Its terms forbid trying to identify speakers and re-hosting or re-sharing the dataset, so
  no audio, sentence or transcript is committed here; reports hold counts and metrics only.
  `client_id` is used only as a grouping key for the split.
- Transcripts come from [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  (MIT) through [mlx-whisper](https://github.com/ml-explore/mlx-examples) (MIT), 0.45–0.57 s per clip on
  an Apple Silicon Mac.

## Split

Linking speakers and sentences puts 300,191 of the 300,315 validated clips into one connected
component, so a split by components is impossible. `make_split.py` takes the official dev / test as
the cores of val / test and leaves out every clip that shares a speaker or a normalized sentence with
them. The official splits share no speaker but do share 52–55 normalized sentences, which are removed.

| Split | Clips | Speakers | Sentences |
| --- | --- | --- | --- |
| train | 147,362 | 3,354 | 20,015 |
| val | 8,965 | 1,142 | 8,940 |
| test | 8,520 | 2,284 | 8,501 |

135,468 validated clips are in no split. The 500 test clips looked at in the trial are excluded from
test. Test is transcribed only for the final evaluation, and `report_asr.py` fails if any test clip
has been transcribed before that.

## Steps

```sh
CV=<dir>/cv-corpus-27.0-2026-09-11/ja   # Common Voice, not in this repository
DATA=<dir>                              # transcripts and manifests, not in this repository
python asr/make_split.py $CV $DATA $DATA/split-v0
export HF_HUB_OFFLINE=1   # the model is cached; no network while transcribing
while :; do python asr/cv_asr.py $CV $DATA/night1.jsonl --order $DATA/split-v0/asr_order.txt --n 156327; [ $? -eq 75 ] || break; done
python asr/report_asr.py $DATA/night1.jsonl $DATA/split-v0/manifest.tsv asr/reports/asr-night1.json asr/PROGRESS.md "<title>"
python asr/classify.py $DATA/night1.jsonl      # verdicts and examples, printed locally only
```

`cv_asr.py` exits with 75 after a clip that does not finish within `--limit` seconds and lists it in
`<out>.hung`; the loop restarts it and the clip is skipped. `nightly.sh <pid> <report command>` waits
for a nightly job, writes its report and commits.
