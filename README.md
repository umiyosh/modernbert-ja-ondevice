# modernbert-ja-ondevice

Run the Japanese masked language models [ModernBERT-Ja](https://huggingface.co/collections/sbintuitions/modernbert-ja) (sbintuitions, MIT) on device. This repository converts them to Core ML so they run on the Apple Neural Engine, and checks every conversion against PyTorch. ONNX for Windows is planned.

The converted model answers one question per row: the log-probability of a token at a masked position. That is all that scoring a word against its alternatives needs (for example, telling 「以外と使いやすい」 from 「意外と使いやすい」), and it keeps the output a single number per row instead of a 102,400-word distribution.

It also has the training code of a Japanese typo detector fine-tuned from modernbert-ja-130m, [umiyosh/modernbert-ja-130m-typo-detector](https://huggingface.co/umiyosh/modernbert-ja-130m-typo-detector) (see [Typo detector](#typo-detector)).

## Results

Apple Silicon Mac, float16, batch 32, measured with `scripts/verify.py`. Δlog P is the difference from PyTorch float32 on real sentences.

| Model | Length | Neural Engine | GPU | CPU | Mean / max Δlog P | Size |
| --- | --- | --- | --- | --- | --- | --- |
| modernbert-ja-30m | 64 | 10.9 ms | 14.2 ms | 21.7 ms | 0.028 / 0.10 | 71 MB |
| modernbert-ja-70m | 64 | 27.5 ms | 32.6 ms | 45.2 ms | 0.031 / 0.17 | 134 MB |
| modernbert-ja-130m | 64 | 53.7 ms | 59.0 ms | 92.1 ms | 0.021 / 0.08 | 253 MB |
| modernbert-ja-130m | 32 | 28.8 ms | 43.8 ms | – | 0.021 / 0.08 | 253 MB |

Time per row depends on the length, not the batch size: batch 128 at length 32 takes 111 ms on the Neural Engine, the same per row as batch 32.

## Usage

```sh
uv venv --python 3.12 && uv pip install -r pyproject.toml
python scripts/convert.py sbintuitions/modernbert-ja-130m build/mbja130m-b32-l32.mlpackage --batch 32 --length 32
python scripts/verify.py sbintuitions/modernbert-ja-130m build/mbja130m-b32-l32.mlpackage --batch 32 --length 32
```

Inputs are `input_ids`, `attention_mask` (batch × length, int32), `position` and `target` (batch, int32); the output `logp` (batch) is log P(target) at position. Pad rows to the length; put the mask token at `position` yourself.

## How the conversion works

- **Encoder without transformers' mask builder.** The model's `forward` builds attention masks with code that does not convert to Core ML. `Scoring.encode` runs the embedding, the encoder layers and the final norm directly with an additive padding mask. The sliding-window layers get the same mask, which is exact while the length is at most `local_attention / 2` (64): every pair of tokens is then inside the window. It matches transformers' output exactly.
- **`torch.export`, not `torch.jit.trace`.** Tracing leaves shape arithmetic as `aten::Int` on tensors that coremltools cannot fold.
- **`gather`, not advanced indexing, for the output.** coremltools turns `logp[rows, target]` into flat offsets (row × 102,400 + id, up to 3.3 million), which float16 cannot hold exactly; in float16 it picked neighbouring tokens (mean Δlog P 0.63, max 25). `gather` along an axis needs no offsets.

- **Running Core ML from Python.** With coremltools 9.0, Core ML's idle reset (`MLE5ExecutionStream resetAfterLingering`) frees coremltools' input bindings on its own queue without the GIL. Under Python's default allocator that races the main thread and crashes in `_PyObject_Free` (exit 139, at a different line each run). Run with `PYTHONMALLOC=malloc`; apps calling Core ML directly are not affected.

## Typo detector

[umiyosh/modernbert-ja-130m-typo-detector](https://huggingface.co/umiyosh/modernbert-ja-130m-typo-detector) marks the tokens of a Japanese sentence that are part of a typo, mainly misconversions (以外と for 意外と). It is modernbert-ja-130m fine-tuned for token classification (`OK` / `TYPO`).

### Training data

| Source | Used for | License |
| --- | --- | --- |
| [sbintuitions/modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m) | Base model | MIT |
| [Japanese Wikipedia Typo Dataset (JWTD) v2](https://nlp.ist.i.kyoto-u.ac.jp/?%E6%97%A5%E6%9C%AC%E8%AA%9EWikipedia%E5%85%A5%E5%8A%9B%E8%AA%A4%E3%82%8A%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88), train split (Kyoto University) | Erroneous sentences and their corrections (1.37 million rows); the corrected sentences and the correction pairs for the synthetic errors | CC BY-SA 3.0 |
| Synthetic errors (`training/augment.py`) | 686,932 corrected JWTD sentences with one error each: a JWTD correction pair reversed (303,234), a homophone (238,225), a word written in katakana as voice input does (96,054), a hiragana word turned into a kanji noun with the same reading (49,419) | Generated from JWTD |
| [UniDic](https://clrd.ninjal.ac.jp/unidic/) via [unidic-lite](https://github.com/polm/unidic-lite) and [fugashi](https://github.com/polm/fugashi) | Readings to generate the synthetic errors; no dictionary data is in the model | BSD-3-Clause (UniDic), MIT |

This repository and the model weights are MIT. The weights contain no JWTD text, and none is redistributed here; the scripts read your own copy of JWTD. Sentences of the evaluation set were excluded from training.

JWTD: 田中佑, 村脇有吾, 河原大輔, 黒橋禎夫. 日本語 Wikipedia の編集履歴に基づく入力誤りデータセットと訂正システムの改良. 言語処理学会第27回年次大会, 2021.

### Reproduce

```sh
uv pip install -r pyproject.toml --extra train
python training/pairs.py jwtd_v2.0/train.jsonl data/pairs.json
python training/build_data.py jwtd_v2.0/train.jsonl data/base.npz --exclude eval-sentences.txt
python training/augment.py jwtd_v2.0/train.jsonl data/pairs.json data/aug --exclude eval-sentences.txt
python training/train.py sbintuitions/modernbert-ja-130m data/base.npz,data/aug.npz ckpt
python scripts/convert_detector.py ckpt/step30000 build/detector-b32-l64.mlpackage --batch 32 --length 64
```

One epoch is 64,332 steps of batch 32 (about 10 hours on an M-series Mac with MPS). The published model is step 30,000, chosen on the development set. `--exclude` takes one sentence per line; with the same exclusions the scripts rebuild the data the model was trained on exactly.

The Core ML detector takes `input_ids` and `attention_mask` (batch × length, int32) and returns `p_typo` (batch × length), P(TYPO) per token: 66 ms for 32 rows of 64 tokens on the Neural Engine.

### Results

Each system marks spans with a score; the threshold is the one that keeps the marks on 60 correct texts (28,148 characters) at one per 1,000 characters or fewer, chosen on the development set (103 items) and applied to the held-out (50) and the fresh (51) sets. Items are misconversions from the JWTD v2 test split and synthetic voice-input errors; the evaluation data is not public (it mixes JWTD text with private texts).

| System | Held-out | Fresh | 3,000 chars |
| --- | --- | --- | --- |
| This detector alone | 52% | 49% | 0.2 s |
| This detector picks the top 15% of words, modernbert-ja-130m scores their candidates | **72%** | **67%** | 0.9 s |
| modernbert-ja-130m scores the candidates of every word | 48% | 55% | 4.1 s |
| [recruit-jp/japanese-typo-detector-roberta-base](https://huggingface.co/recruit-jp/japanese-typo-detector-roberta-base) alone, 1 − P(OK) | 46% | 43% | – |

Times are Core ML on an Apple Silicon Mac. The candidates come from homophones and the JWTD correction pairs; scoring compares the masked-LM likelihood of the written word and each candidate in the sentence. The recruit-jp model detects every kind of typo, while these items are misconversions only. On the development and held-out sets together (153 items), screened scoring finds 70%, scoring every word 47% and the recruit-jp model 47% (paired exact test p < 0.001 for both). Screening helps accuracy as well as time: scoring every word also scores the correct words whose alternatives happen to fit, and those false alarms force a higher threshold.

Limits: JWTD is written in full-width characters with no line breaks, so half-width runs and line breaks can draw marks; feed the detector one sentence at a time. At this strict false-alarm rate the detector alone finds about half of the misconversions; it works best as the first stage of candidate scoring.

## Voice-input errors

The models published now come from the work on issue #98 of Kotobae, a proofreading app (its repository is private): speech recognition writes a word that sounds like the one said (誤変換 heard as 5変換), and the detector above, trained on typed text, missed most of them. The code, the evaluation protocol written before training and the reports are in [`asr/`](asr/README.md).

- [umiyosh/modernbert-ja-130m-typo-detector](https://huggingface.co/umiyosh/modernbert-ja-130m-typo-detector) rev `33115ea`: the detector above, trained further on recognition errors (`asr/train.py`), pre-trained on synthetic kana-kanji conversion confusions (`asr/synth_kkc.py`), and averaged with the earlier weights (`asr/soup.py`)
- [umiyosh/modernbert-ja-130m-homophone-scorer](https://huggingface.co/umiyosh/modernbert-ja-130m-homophone-scorer) rev `0abe0ed`: modernbert-ja-130m trained to put the word that was said above the homophones speech recognition writes for it (`asr/scorer_pairs.py`, `asr/train_scorer.py`)

The model cards hold the training settings in full.

### Data

| Source | Used for | License |
| --- | --- | --- |
| [Mozilla Common Voice](https://commonvoice.mozilla.org/) 27.0 Japanese | Recognition errors for training (train split) and the evaluation (val, test); split by speaker and sentence with no overlap (`asr/make_split.py`, `asr/leak_check.py`) | CC0; its terms forbid re-sharing the dataset and identifying speakers, so no audio, sentence or transcript is in this repository |
| [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) through [mlx-whisper](https://github.com/ml-explore/mlx-examples) | The transcripts the errors come from | MIT |
| [reazon-research/reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) | Second recognizer: a word both recognizers hear the same way is taken as the reader's slip and left out (`asr/labels.py`) | Apache-2.0 |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) as [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) | Readings of corrected JWTD sentences, transcribed for more errors (`asr/synth_tts.py`; only Kokoro's readings train the published models) | Apache-2.0 |
| [yuuki14202028/kanakanji-edge-33m](https://huggingface.co/yuuki14202028/kanakanji-edge-33m) | Synthetic conversion confusions | Apache-2.0 |
| JWTD v2 (see above) | Sentences and pairs as above; not redistributed | CC BY-SA 3.0 |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs) ja_jp | External holdout only (`asr/fleurs.py`) | CC BY 4.0 |

### Results

Measured once on Common Voice test after every choice was fixed ([`asr/PROTOCOL.md`](asr/PROTOCOL.md), [`asr/reports/final-test.json`](asr/reports/final-test.json)): 6,563 clips of 2,091 speakers, 523 homophone errors, each system at its fixed point, run through Kotobae's own pipeline on Core ML. Intervals are 95%, by bootstrap over speakers.

| System | Homophone recall | All error spans | False marks per 1,000 characters of correct transcripts |
| --- | --- | --- | --- |
| Published detector and scorer (0.995 / 7.0) | **66.3%** (61.9–70.6) | 21.5% | 1.05 (0.80–1.32) |
| The detector above with modernbert-ja-130m scoring (before #98) | 22.0% | 8.8% | 0.88 |
| Fujii and Shinnou (2021) zero-shot BERT scoring, same candidates | 41.3% | 19.2% | 0.98 |
| Jev, a commercial per-phrase error check | 23.7% | 28.8% | 1.89 |
| Rules alone (confusion table, unknown words) | 58.1% | 30.2% | 60.35 |

Against the version before #98 the gain is +44.4 points in homophone recall and +12.7 points in all error spans (p < 0.001 for both); the false-mark difference (+0.18) is not significant (p = 0.25). The goal was 60% homophone recall at about one false mark per 1,000 characters. Before test the judgment was written as at most 1.00; at 1.05 it misses that line by 0.05, and Kotobae's owner counts the goal as met.

On FLEURS, transcribed with ReazonSpeech instead of Whisper and made of written sentences, homophone recall is 42.9% (35.1–50.8) at 0.42 false marks, against 22.3% before #98 ([`asr/reports/final-fleurs.json`](asr/reports/final-fleurs.json)): the gain carries over, the level does not.

Limits: the check reads text only, so a wrong word that also fits the sentence (自己 for 事故) cannot be told apart without the audio.

## License

MIT. The converted weights come from sbintuitions/modernbert-ja-*, also MIT. The typo detector's weights are MIT; see [Training data](#training-data) for the sources and their licenses.
