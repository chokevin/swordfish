# CuTe/CUTLASS GEMM spike

This directory is the native-extension slot for a future CuTe/CUTLASS GEMM.
The runner already accepts `--backend cutlass`; it intentionally fails with an
explicit environment error until the optional extension module is built.

Expected Linux CUDA build flow:

```bash
export CUTLASS_DIR=/path/to/cutlass
python -m swordfish.kernels.cute.build --cutlass-dir "$CUTLASS_DIR"
python -m swordfish.runner run-gemm --backend cutlass --m 4096 --n 4096 --k 4096 --dtype fp16 --device auto --out /tmp/cutlass-gemm.json
```

Do not make this backend fall back to `torch.mm`; correctness and performance
comparisons are only useful when the reported backend is the backend that ran.
