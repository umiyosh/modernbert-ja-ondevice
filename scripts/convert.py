"""Convert a ModernBERT-Ja masked LM to a Core ML scoring model.

The converted model returns only what scoring needs: log P(target) at one masked position per
row. Shapes are fixed (batch x length) because the Neural Engine runs fixed shapes; callers pad
rows to the length.

usage: python scripts/convert.py <hf model> <out.mlpackage> [--batch 32] [--length 64]
"""
import argparse

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from transformers import AutoModelForMaskedLM


# coremltools 9.0 has no converter for Tensor.new_ones. transformers' mask builder calls it as
# new_ones((), dtype=torch.bool), the identity for and-ing mask functions.
@register_torch_op
def new_ones(context, node):
    shape = _get_inputs(context, node)[1]
    if shape.val is not None and np.size(shape.val) == 0:
        context.add(mb.const(val=True, name=node.name))
    else:
        context.add(mb.fill(shape=mb.cast(x=shape, dtype="int32"), value=1.0, name=node.name))


class Scoring(torch.nn.Module):
    """log P(target | sequence) at `position` for each row, from the masked LM's head.

    Runs the encoder layers directly with an additive padding mask instead of transformers' mask
    builder, which does not trace to Core ML. The sliding-window layers get the same mask: every
    pair of tokens lies within the local window when the length is at most local_attention / 2.
    """

    def __init__(self, mlm):
        super().__init__()
        self.mlm = mlm

    def encode(self, input_ids, attention_mask):
        m = self.mlm.model
        hidden = m.embeddings(input_ids=input_ids)
        mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * -1e4
        ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        rope = {kind: m.rotary_emb(hidden, ids, kind) for kind in set(m.config.layer_types)}
        for layer in m.layers:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=rope[layer.attention_type])
        return m.final_norm(hidden)

    def forward(self, input_ids, attention_mask, position, target):
        hidden = self.encode(input_ids, attention_mask)
        # gather along an axis instead of advanced indexing: the converter turns advanced
        # indexing into flat offsets (row * vocab + id, up to 3.3 million), which float16
        # cannot hold exactly, so it picked neighbouring tokens.
        at = position[:, None, None].expand(-1, 1, hidden.shape[-1])
        picked = hidden.gather(1, at)[:, 0]
        logp = torch.log_softmax(self.mlm.decoder(self.mlm.head(picked)), dim=-1)
        return logp.gather(1, target[:, None])[:, 0]


def precision(a):
    if a.fp32:
        return ct.precision.FLOAT32
    keep = {x for x in a.fp32_ops.split(",") if x}
    if not keep:
        return ct.precision.FLOAT16
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--fp32", action="store_true", help="compute in float32 (CPU/GPU only; for checking)")
    p.add_argument("--fp32-ops", default="", help="comma-separated MIL op types kept in float32")
    a = p.parse_args()
    mlm = AutoModelForMaskedLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
    model = Scoring(mlm).eval()
    b, n = a.batch, a.length
    if n > mlm.config.local_attention // 2:
        raise SystemExit(f"--length must be at most {mlm.config.local_attention // 2} so one mask serves every layer")
    example = (torch.randint(5, 1000, (b, n)), torch.ones(b, n, dtype=torch.long),
               torch.zeros(b, dtype=torch.long), torch.zeros(b, dtype=torch.long))
    # torch.export keeps shape arithmetic static; jit.trace leaves aten::Int on tensors that
    # coremltools cannot fold.
    with torch.no_grad():
        exported = torch.export.export(model, example).run_decompositions({})
    mlmodel = ct.convert(
        exported,
        inputs=[ct.TensorType("input_ids", shape=(b, n), dtype=np.int32),
                ct.TensorType("attention_mask", shape=(b, n), dtype=np.int32),
                ct.TensorType("position", shape=(b,), dtype=np.int32),
                ct.TensorType("target", shape=(b,), dtype=np.int32)],
        outputs=[ct.TensorType("logp", dtype=np.float32)],
        compute_precision=precision(a),
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
