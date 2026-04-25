# Autoresearch run `20260420T014943Z`

- **source SHA:** `20ab7f3`
- **GPU:** NVIDIA A100-SXM4-80GB (cc 8.0, 79.3 GB)
- **CUDA / torch / triton:** 12.4 / 2.4.0a0+07cecf4168.nv24.05 / 3.0.0
- **shapes:** `voice`  **impls:** `fp16,marlin`  **repeats:** 5
- **marlin SHA:** `1f25790bdd49fba53106164a24666dade68d7c90`

## Results

| shape | impl | ms_mean | ms_p95 | TFLOPS | speedup vs fp16 | error |
|---|---|---|---|---|---|---|
| 8b-b1 | fp16 | 0.031 | 0.033 | 1.1 | x1.00 | |
| 8b-b1 | marlin | 0.049 | 0.051 | 0.7 | x0.64 | |
| 8b-b4 | fp16 | 0.031 | 0.031 | 4.4 | x1.00 | |
| 8b-b4 | marlin | 0.050 | 0.050 | 2.7 | x0.61 | |
| 8b-b8 | fp16 | 0.031 | 0.032 | 8.6 | x1.00 | |
| 8b-b8 | marlin | 0.050 | 0.050 | 5.4 | x0.63 | |
| 70b-tp2-b1 | fp16 | 0.051 | 0.056 | 1.3 | x1.00 | |
| 70b-tp2-b1 | marlin | 0.049 | 0.050 | 1.4 | x1.02 | |
| 70b-tp2-b4 | fp16 | 0.049 | 0.050 | 5.4 | x1.00 | |
| 70b-tp2-b4 | marlin | 0.066 | 0.133 | 4.1 | x0.75 | |
| 70b-tp2-b8 | fp16 | 0.049 | 0.050 | 10.8 | x1.00 | |
| 70b-tp2-b8 | marlin | 0.049 | 0.049 | 10.9 | x1.01 | |

![roofline](./roofline.png)
