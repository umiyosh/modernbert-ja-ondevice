#!/bin/sh
# Kotobae #98 step 1: the scorer (modernbert-ja-130m) fine-tuned on voice-input pairs (train_scorer.py),
# then each checkpoint and the base model rescore the shipped run's candidates (pll.py, the app's own
# scoring) on val, Kotobae's texts and the Kokoro held-out set, and prod.py picks thresholds under the
# three conditions. Everything under guard.py, one at a time.
set -u
D=$HOME/dvlp/kotobae/.jev-eval/typo-study/asr
PY=$HOME/dvlp/kotobae/.jev-eval/typo-study/.venv/bin/python
R=$HOME/dvlp/modernbert-ja-ondevice-asr/asr
G="python3 $D/../guard.py --max-gb 12 --min-free 30 --"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd $D
echo "$(date +%H:%M) train"
$G $PY $R/train_scorer.py scorer/pairs.jsonl scorer/run1 --steps 6000 --save-every 1500 > scorer/run1.log 2>&1 || echo "train failed"
for m in base step1500 step3000 step4500 step6000; do
  if [ $m = base ]; then M=sbintuitions/modernbert-ja-130m; else M=scorer/run1/$m; fi
  n=sc1$m
  echo "$(date +%H:%M) rescore $n"
  $G $PY tools/pll.py eval-val ship1 labels.jsonl $M $n > scorer/pll-$n-val.log 2>&1 || echo "$n val failed"
  $G $PY tools/pll.py .. ship1 ../texts.json $M $n > scorer/pll-$n-kotobae.log 2>&1 || echo "$n kotobae failed"
  $G $PY tools/pll.py eval-kokoro ship1 eval-kokoro/labels.jsonl $M $n > scorer/pll-$n-synth.log 2>&1 || echo "$n synth failed"
done
echo "$(date +%H:%M) choose"
$PY tools/prod.py --gate ship1:ship1:ship1 sc1base:sc1base:sc1base sc1step1500:sc1step1500:sc1step1500 sc1step3000:sc1step3000:sc1step3000 sc1step4500:sc1step4500:sc1step4500 sc1step6000:sc1step6000:sc1step6000 > scorer/run1-prod.txt 2>&1
echo "$(date +%H:%M) done"
