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
- 2026-10-19 labels v1 (reports/labels-v1.json): every train and val clip of split v1 has both transcripts. Train 142,143 clips: 89,369 clean, 34,231 error (10,896 homophone edits), 18,543 quarantined (14,850 reader slips, 2,170 reported, 2,106 broken, 10 short). Val unchanged from the baseline run (4,473 clean, 2,416 error, 1,825 quarantined). Broken transcripts are 5.9% of val but 1.5% of train; to be checked before training.
- 2026-10-19 broken-transcript gap (val 5.9%, train 1.5%) checked before training: not a few speakers (without the top 10 val speakers, 5.1%), not speaker volume (train 2.0% vs val 5.7% among speakers with 21–100 clips), not script length (val 3–5x train in every length band) and not Whisper having seen the official train split (official train clips 2.5% broken, 37.4% with errors, worse than the other validated clips at 1.3%, 26.3%). The gap follows Common Voice's official splits: val (official dev) is harder for Whisper than the repeated-sentence clips that make up most of train. Broken clips leave training and every metric; val and test are both official splits, so model selection on val carries to test. Cause left open; test's broken rate is reported at step 9.
- 2026-10-19 baseline A on val (reports/val-A.json): the rules alone (spans the misconversion table lists and kanji / katakana runs IPADIC does not know, written by Kotobae's typo_eval as rules-<name>.json) find 26.2% of error spans (95% CI 24.6–27.8) and 53.2% of homophone spans (48.9–57.2), at 60.3 false marks per 1,000 characters of clean clips (58.7–62.1), 60 times the budget. A has no threshold, so it has no operating point; it shows how much the rules reach, not a usable system. The same run reproduced D's probes and candidate scores exactly.
- 2026-10-19 step 7, seed 0, val only (selection numbers, optimistic): fine-tuning D's detector on speech-error rows alone (lr 2e-5, 2 epochs) lifts detector-only span recall at the 1-per-1,000 budget from 6.3% to 10.2% (step 2,000 of 3,488). Mixing the rows 1:1 with JWTD replay keeps it at D's level (5.1–6.7%), so that mix is dropped; whether speech-only training loses JWTD-type typos is checked on Kotobae's sets before test. In Kotobae's pipeline (Core ML, candidate scoring), the step-2,000 detector with thresholds refit on val (detector 0.91, score 9.0) reaches span recall 13.2% and homophone recall 28.6% at 0.94 false marks per 1,000 characters, against D's 8.2% and 21.9% at its shipped thresholds (1.18). The sharper detector sends twice as many words to candidate scoring (8,279 vs 4,122).

## 2026-10-27: final evaluation

- Amendment before test (PROTOCOL.md): C fixed as the check shipped in Kotobae (PR #121) at detector 0.995 / score 7.0; the kotobae#98 goal added as its own judgment; FLEURS ja as the external holdout. Test unsealed at 18:01 and transcribed once (8,520 clips with Whisper and ReazonSpeech, none hung).
- Test labels (reports/labels-test.json): 2,155 error, 4,478 clean, 1,887 quarantined (794 broken, 1,045 reader slips, 79 reported, 2 short); 525 homophone edits. 6,563 clips of 2,091 speakers count; 70 JWTD clips are reported apart.
- Test, each system at its fixed point (reports/final-test.json; 95% intervals by speaker bootstrap):
  - C: span recall 21.5% (19.8–23.2), homophone recall 66.3% (61.9–70.6; 347 of 523), 1.05 false marks per 1,000 characters of clean clips (0.80–1.32), precision 94%
  - D: 8.8% (7.8–9.8), 22.0% (18.5–25.5), 0.88 (0.64–1.13), 81%
  - A (rules alone, no operating point): 30.2%, 58.1%, 60.35 false marks per 1,000 characters, 33%
  - B (Jev, 6,635 requests): 28.8%, 23.7%, 1.89, 86%
  - F (Fujii & Shinnou 2021 on C's candidates, reference only): 19.2%, 41.3% (36.9–45.7), 0.98
- Primary test: C − D in span recall +12.7 points (11.1–14.3), p < 0.001. Secondary (Holm): homophone recall +44.4 points (39.0–49.7), p < 0.001; false marks +0.18 per 1,000 characters (−0.10 to +0.45), p = 0.25. C − B in homophone recall +42.6 points, C − F +25.1 points (both p < 0.001), C − A +8.2 points (p = 0.007).
- External holdout, FLEURS ja (reports/final-fleurs.json; 2,638 clips, 1,574 sentences, 238 homophone edits; ReazonSpeech transcripts; resampled by sentence): C span recall 17.6% (14.6–20.7), homophone 42.9% (35.1–50.8), 0.42 false marks; D 9.1%, 22.3%, 0.33; B 31.6%, 26.9%, 3.63; F 16.5%, 30.3%, 0.32. C − D span recall +8.4 points (5.8–11.2).
- Decision (three axes of PROTOCOL.md): statistical, the interval of C − D excludes 0; practical, +12.7 ≥ +5 points with 0.50 false marks per 1,000 characters on the 60 Kotobae texts; generalization, C ≥ D on FLEURS within the budget. **Adopt.**
- kotobae#98 goal: homophone recall reaches 66.3% with the interval's lower end at 61.9%, but false marks on test clean clips are 1.05 per 1,000 characters, over the condition of at most 1.00 written before test; by the rule as written, neither "reached" nor "shown" holds. The false-mark rate is not significantly above D's (+0.18, p = 0.25) and its interval includes 1.00. On FLEURS, homophone recall is 42.9%: the 60% does not carry to another recognizer and to written text.
- 2026-10-27, after the results: the product owner counts the kotobae#98 goal as met. The goal was "about 1 false mark per 1,000 characters"; 1.05 is within it and its interval (0.80–1.32) includes 1.00. The judgment written before test (at most 1.00, point estimate) is left as it was: by that rule neither "reached" nor "shown" holds, by 0.05.
