# modernbert-ja-ondevice

Run the Japanese masked language models [ModernBERT-Ja](https://huggingface.co/collections/sbintuitions/modernbert-ja) (sbintuitions, MIT) on device. This repository converts them to Core ML so they run on the Apple Neural Engine, and checks every conversion against PyTorch. ONNX for Windows is planned.

The converted model answers one question per row: the log-probability of a token at a masked position. That is all that scoring a word against its alternatives needs (for example, telling 「以外と使いやすい」 from 「意外と使いやすい」), and it keeps the output a single number per row instead of a 102,400-word distribution.

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

## License

MIT. The converted weights come from sbintuitions/modernbert-ja-*, also MIT.
