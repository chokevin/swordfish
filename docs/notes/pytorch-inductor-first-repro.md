# PyTorch/Inductor first repro candidate

## Candidate

Use the GPT-style decoder block as a tiny `torch.compile` repro focused on
causal-mask slicing, `masked_fill(..., -inf)`, matmul, fp32 softmax, and residual
MLP composition.

This is a good first Inductor candidate because the surface is realistic but
small: one block, fixed shapes, no training loop, no tokenizer, no distributed
state.

## Minimal script sketch

```python
import torch

from swordfish.transformer.config import tiny_test_config
from swordfish.transformer.model import GPTDecoderBlock

torch.manual_seed(0)

cfg = tiny_test_config()
model = GPTDecoderBlock(cfg).eval().to("cuda", dtype=torch.float16)
x = torch.randn(2, 16, cfg.n_embd, device="cuda", dtype=torch.float16)

with torch.no_grad():
    eager = model(x)
    compiled_model = torch.compile(model, fullgraph=True)
    compiled = compiled_model(x)

diff = (eager - compiled).abs()
print("max_abs_diff", diff.max().item())
print("eager_finite", eager.isfinite().all().item())
print("compiled_finite", compiled.isfinite().all().item())
print("eager_checksum_fp32", eager.float().sum().item())
print("compiled_checksum_fp32", compiled.float().sum().item())
```

## What to compare

- eager vs compiled maximum absolute error,
- finite-output status for both paths,
- fp32 checksum for both paths,
- compile/runtime exception if there is one,
- GPU, CUDA, driver, PyTorch version, and source commit.

## Draft issue shape

Title:

> `torch.compile` repro for causal-mask `masked_fill(-inf)` GPT block

Only open this as an issue if there is a real failure, excessive compile-time
failure, graph break under the intended mode, or unexpected numerical behavior.
If eager and compiled match cleanly, keep it as a local regression script for
future transformer-kernel work instead.

## Caveats

Small fp16 numerical drift is not a compiler bug by itself. Use `eval()` and a
fixed seed so dropout does not create false differences.
