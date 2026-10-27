#!/bin/sh
# Kotobae #98 step 2b: the scorer trained again from modernbert-ja-130m with margins set per kind (a fix
# pair should clear the app's mark, 6.0; a keep or hard pair only has to keep the written word ahead, 1.0)
# and the hard negatives mined from correct training-side text (scorer/hard.jsonl, twice). The shipped
# scorer (run1/step3000) and each checkpoint rescore the shipped run's candidates on all four sets
# (pll.py), and recalibrate.py picks thresholds on the dev half under every false-mark limit.
# The swap generation (step 2c's data) pauses for this and resumes afterwards.
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
PY=$HOME/dvlp/kotobae/.jev-eval/typo-study/.venv/bin/python
R=$HOME/dvlp/modernbert-ja-ondevice-asr/asr
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd $D
pkill -f "kkc/night1.sh"; pkill -f "synth_kkc.py ../jwtd"; sleep 10
echo "$(date +%H:%M) generation paused at $(cat kkc/pretrain.jsonl.done)"
echo "$(date +%H:%M) train"
python3 ../guard.py --max-gb 12 --min-free 30 -- $PY $R/train_scorer.py scorer/pairs.jsonl,scorer/hard.jsonl,scorer/hard.jsonl scorer/run2 \
  --steps 6000 --save-every 1500 --margin-fix 6.0 --margin-keep 1.0 > scorer/run2.log 2>&1 || echo "train failed"
for m in base step1500 step3000 step4500 step6000; do
  if [ $m = base ]; then M=scorer/run1/step3000; else M=scorer/run2/$m; fi
  n=sc2$m
  echo "$(date +%H:%M) rescore $n"
  for e in "eval-val labels.jsonl" ".. ../texts.json" "eval-kokoro eval-kokoro/labels.jsonl" "eval-clean eval-clean/texts.json"; do
    set -- $e
    python3 ../guard.py --max-gb 8 --min-free 30 -- $PY tools/pll.py $1 ship2 $2 $M $n > scorer/pll-$n-$(basename $1).log 2>&1 || echo "$n $1 failed"
  done
  $PY tools/recalibrate.py $n > scorer/recal-$n.txt 2>&1
done
echo "$(date +%H:%M) rescored; generation resumes"
sh kkc/night1.sh > kkc/night1-main.log 2>&1
echo "$(date +%H:%M) done"
