# Swordfish benchmark completion report

**Status:** READY

## Gate configuration

- Result directory: `runs/airun/week1`
- Architectures: `a100, h100, h200`
- Backend: `torch`
- Prefix: `torch-gemm`
- Dtype: `fp16`
- Shape: `m=4096 n=4096 k=4096`
- Recursive search: `True`
- Require complete NCU: `True`

## Completion gate

- Complete: every requested architecture has a valid matching result.

## Indexed artifacts

- Result rows: `5`
- Skipped JSON files: `7`

| file | benchmark | backend | gpu | dtype | shape | mean_ms | tflops | matches_reference | ncu_complete | protocol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w4a16-triton-a100.json | marlin_w4a16_matmul | triton | a100 | fp16 | m=64 n=64 k=64 | 0.062048 | 0.00844972 | True |  | OK |
| w4a16-triton-h100.json | marlin_w4a16_matmul | triton | h100 | fp16 | m=64 n=64 k=64 | 0.0323296 | 0.016217 | True |  | OK |
| torch-gemm-a100.json | torch_gemm | torch | a100 | fp16 | m=4096 n=4096 k=4096 | 0.604938 | 227.195 | True | True | OK |
| torch-gemm-h100.json | torch_gemm | torch | h100 | fp16 | m=4096 n=4096 k=4096 | 0.277896 | 494.57 | True | True | OK |
| torch-gemm-h200.json | torch_gemm | torch | h200 | fp16 | m=4096 n=4096 k=4096 | 0.182761 | 752.014 | True | True | OK |
