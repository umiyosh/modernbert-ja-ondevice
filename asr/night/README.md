# Overnight drivers for Kotobae #98

The shell scripts that ran each overnight step, kept as they ran (from `typo-study/asr`, where the data
lives). Every heavy job goes through `../guard.py`.

- `ship-eval.sh`, `ship3-eval.sh`: the app pipeline (Rust, Core ML) on the evaluation sets
- `scorer-1.sh`, `scorer-2b.sh`: scorer fine-tuning and rescoring (steps 1 and 2b)
- `clean-eval.sh`, `mine-eval.sh`: correct text for false marks, and hard-negative mining
- `swap-generation.sh`: IME-simulated swaps for step 2c, restarted from where a guard stop left it
