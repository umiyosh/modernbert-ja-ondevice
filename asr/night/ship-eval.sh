#!/bin/sh
# The app's own pipeline as it will ship (#98): detector kaw03, the voice-input table merged into
# the misconversion table, voice words forced to the scorer, marks at 0.6 / 6.0. Writes the same
# files as screen0.sh so prod.py can confirm the offline policy numbers. usage: sh ship-eval.sh <name>
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
W=$HOME/dvlp/kotobae/.wt/feat/asr-detector/src-tauri
N=$1
for e in "$D/eval-val val" "$D/.. kotobae" "$D/eval-kokoro synth"; do
  set -- $e $N
  echo "$(date +%H:%M) $3 $2"
  (cd $W && CARGO_TARGET_DIR=$HOME/dvlp/kotobae/src-tauri/target KOTOBAE_TYPO_EVAL=$1 KOTOBAE_TYPO_EVAL_NAME=$3 \
    cargo test --release --lib services::typo::eval_tests::typo_eval -- --ignored --exact --nocapture > $D/coreml/log-$3-$2.txt 2>&1) || echo "$3 $2 failed"
done
echo "$(date +%H:%M) done"
