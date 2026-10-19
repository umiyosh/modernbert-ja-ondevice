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
- 2026-10-18 quality check (reports/quality-v0.json): no empty clip, none over 30 s, no audio hash shared within or across splits. 13 clips are under 1 s, and 2,357 train / 129 val / 79 test clips read a sentence reported on Common Voice (mostly grammar or spelling, other language, hard to pronounce); both are quarantined when labels are made, since the script may not be the truth.
- 2026-10-18 split v1 (reports/near-dup-v1.json): 289 val and 305 test sentences have a train sentence at character 3-gram Jaccard 0.5 or more (often the same frame with one word swapped). The 542 such train sentences (5,215 clips) are removed from train; val and test are unchanged. Train v1: 142,147 clips.
- 2026-10-18 leak check (reports/leak-v1.json): split v1 rebuilt with all-pairs near-duplicate search (the first candidate search missed one train-val pair); val also loses 245 sentences near test. Train 142,143, val 8,714, test 8,520 clips; no raw, normalized or near sentence, speaker or audio hash shared by any two splits; no test clip transcribed. 94 val and 80 test sentences are in JWTD train and leave the primary metric. manifest.tsv and test.tsv are sealed (no read permission) until step 9.
- 2026-10-19 baseline D on val (reports/val-D.json; labels quarantine 1,825 of 8,714 val clips, 1,223 of them reader slips heard alike by Whisper and ReazonSpeech): span recall 8.2% (95% CI 7.3–9.2, 3,409 spans, 1,114 speakers), homophone recall 21.9% (18.4–25.6), 1.18 false marks per 1,000 characters of clean clips (0.90–1.47), precision 81%.
- 2026-10-19 00:50 resume transcribing Common Voice ja with the hang guard: train 62,077 clips (36.5% with errors, 5,833 homophone edits); val 8,965 clips (41.8% with errors, 739 homophone edits). Test transcripts: 0.
- 2026-10-19 13:14 transcribe the rest of Common Voice ja train: train 147,362 clips (34.6% with errors, 13,215 homophone edits); val 8,965 clips (41.8% with errors, 739 homophone edits). Test transcripts: 0.
