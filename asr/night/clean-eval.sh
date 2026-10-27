#!/bin/sh
# False marks on 150,000 characters of correct written text (JWTD test's corrected sentences, outside the
# typo sets) for the check before #98 (the main checkout: its code and models), the #117 version and the
# #118 version, so the false-mark limit rests on more text than the 72,000 characters of clean Common Voice.
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
G="python3 $D/../guard.py --max-gb 6 --min-free 30 --shared --"
for e in "$HOME/dvlp/kotobae/src-tauri d" "$HOME/dvlp/kotobae/.wt/feat/asr-detector/src-tauri ship1" "$HOME/dvlp/kotobae/.wt/feat/homophone-scorer/src-tauri ship2"; do
  set -- $e
  echo "$(date +%H:%M) $2"
  (cd $1 && CARGO_TARGET_DIR=$HOME/dvlp/kotobae/src-tauri/target KOTOBAE_TYPO_EVAL=$D/eval-clean KOTOBAE_TYPO_EVAL_NAME=$2 \
    $G cargo test --release --lib services::typo::eval_tests::typo_eval -- --ignored --exact --nocapture > $D/coreml/log-clean-$2.txt 2>&1) || echo "$2 failed"
done
echo "$(date +%H:%M) done"
