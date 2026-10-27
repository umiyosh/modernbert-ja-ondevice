#!/bin/sh
# The app's pipeline with the 2b scorer (run2/step6000) in place of the shipped one, on all four sets,
# to confirm the offline numbers (recalibrate.py sc2step6000) before publishing it. Core ML on the Neural
# Engine beside the swap generation on the GPU, so the guard runs it shared with its own cap.
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
W=$HOME/dvlp/kotobae/.wt/feat/homophone-scorer/src-tauri
for e in "$D/eval-val val" "$D/.. kotobae" "$D/eval-kokoro synth" "$D/eval-clean clean"; do
  set -- $e
  echo "$(date +%H:%M) $2"
  (cd $W && CARGO_TARGET_DIR=$HOME/dvlp/kotobae/src-tauri/target KOTOBAE_TYPO_EVAL=$1 KOTOBAE_TYPO_EVAL_NAME=ship3 \
    python3 $D/../guard.py --max-gb 6 --min-free 30 --shared -- cargo test --release --lib services::typo::eval_tests::typo_eval -- --ignored --exact --nocapture > $D/coreml/log-ship3-$2.txt 2>&1) || echo "$2 failed"
done
echo "$(date +%H:%M) done"
