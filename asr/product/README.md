# Product yardsticks for Kotobae's on-device misconversion check

Scripts run from `typo-study/asr` (linked there as `tools/`) on the outputs of the app's own pipeline
(`typo/eval_tests.rs`: `res-`, `deltas-`, `cands-` files per run). They read Common Voice, Kokoro and
Kotobae evaluation files that stay outside this repository; only counts and metrics are printed.

- `prod.py`: the three product conditions for one run (voice-input homophone recall on val and the Kokoro
  held-out readings, typo sets found per subset, false marks per 1,000 characters of correct text), at
  the thresholds that keep false marks within the published check's
- `prodforce.py`, `learned.py`, `curve.py`: screening and marking policies chosen offline on a run that
  scored every candidate
- `pll.py`, `kk_score.py`, `clm.py`: rescore a run's candidates with another model, as the app scores them
- `stages.py`, `halves.py`: where each real homophone is lost, and dev/calibration halves by speaker
- `voice_table.py`: the voice-input confusion table
- the rest are one-off probes kept for the record of #98
