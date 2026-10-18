# Progress

Numbers come from train and val only until the final evaluation; test stays sealed.

## 2026-10-17

- Downloaded Common Voice Scripted Speech 27.0 Japanese (15,417,014,005 bytes, SHA-256 matches the
  catalogue). 585,330 clips, 300,315 validated, 37,502 normalized sentences.
- Whisper (MLX) runs 0.45 s per clip and gave the same text as the transformers pipeline on 50 of 50
  clips synthesized with macOS `say` (not Common Voice).
- Trial on 500 clips from the official test and 500 from train: 40–44% of clips have errors, 7–10
  homophone errors per 100 clips, no sign that Whisper remembers the train clips. The rules of
  `classify.py` were set while reading these clips, so the 500 test clips are excluded from test.
- Split v0: no clip, speaker or normalized sentence crosses train / val / test (see README).
- Evaluation protocol fixed before training: span recall at ≤1 false mark per 1,000 characters, C
  (new detector) against D (Kotobae today), speaker-level paired bootstrap and permutation test, and
  adopt / reject / undecided on three separate axes (PROTOCOL.md).
- 2026-10-18 10:01 transcribe Common Voice ja for night 1: train 15,849 clips (36.7% with errors, 1,497 homophone edits); val 8,965 clips (41.8% with errors, 739 homophone edits). Test transcripts: 0.
