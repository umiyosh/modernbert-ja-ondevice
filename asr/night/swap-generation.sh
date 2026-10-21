#!/bin/sh
# Kotobae #98 step 2c (data): IME-simulated misconversions for the detector's pretraining
# (synth_kkc.py, CSCD-NS recipe), up to 6 words per sentence, under guard.py. A run the guard stops
# (memory cap, low system memory) is started again from kkc/pretrain.jsonl.done, which frees its memory;
# everything stops at 09:20 when the Mac goes back to work.
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
PY=$HOME/dvlp/kotobae/.jev-eval/typo-study/.venv/bin/python
R=$HOME/dvlp/modernbert-ja-ondevice-asr/asr
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 KKC_BATCH=64
cd $D
late() { t=$(date +%H%M); [ "$t" -ge 0920 ] && [ "$t" -lt 1200 ]; }
for n in 1 2 3 4 5 6 7 8 9 10; do
  late && break
  echo "$(date +%H:%M) run $n from $(cat kkc/pretrain.jsonl.done 2>/dev/null || echo 0)"
  python3 ../guard.py --max-gb 10 --min-free 30 -- $PY $R/synth_kkc.py ../jwtd/jwtd_v2.0/train.jsonl kkc/pretrain.jsonl --per 6 >> kkc/night1.log 2>&1 &
  G=$!
  while kill -0 $G 2>/dev/null; do
    late && { echo "$(date +%H:%M) stop for the day"; pkill -f "synth_kkc.py ../jwtd"; }
    sleep 60
  done
  wait $G; code=$?
  echo "$(date +%H:%M) run $n exit $code"
  [ $code -eq 0 ] && break
  sleep 120
done
echo "$(date +%H:%M) end, kept $(wc -l < kkc/pretrain.jsonl), done $(cat kkc/pretrain.jsonl.done)"
