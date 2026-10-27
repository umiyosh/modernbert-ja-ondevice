#!/bin/sh
# Kotobae #98 step 2b: the shipped pipeline (#118 models) over 30,000 correct training-side sentences, to
# find the alternatives the scorer favours on correct text (hard negatives). Core ML on the Neural Engine,
# beside the swap generation on the GPU, so the guard runs it shared with its own cap.
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
cd $HOME/dvlp/kotobae/.wt/feat/homophone-scorer/src-tauri && echo "$(date +%H:%M) mine" && \
  CARGO_TARGET_DIR=$HOME/dvlp/kotobae/src-tauri/target KOTOBAE_TYPO_EVAL=$D/eval-mine KOTOBAE_TYPO_EVAL_NAME=ship2 \
  python3 $D/../guard.py --max-gb 6 --min-free 30 --shared -- cargo test --release --lib services::typo::eval_tests::typo_eval -- --ignored --exact --nocapture > $D/coreml/log-mine-ship2.txt 2>&1 || echo "mine failed"
echo "$(date +%H:%M) done"
