# Autoresearch run `20260420T015907Z`

- **source SHA:** `d212cd6`
- **GPU:** NVIDIA A100-SXM4-80GB (cc 8.0, 79.3 GB)
- **CUDA / torch / triton:** 12.4 / 2.4.0a0+07cecf4168.nv24.05 / 3.0.0
- **shapes:** `voice`  **impls:** `fp16,marlin`  **repeats:** 5
- **marlin SHA:** `1f25790bdd49fba53106164a24666dade68d7c90`

## Results

| shape | impl | ms_mean | ms_p95 | TFLOPS | speedup vs fp16 | error |
|---|---|---|---|---|---|---|
| 8b-b1 | fp16 | 0.032 | 0.034 | 1.1 | x1.00 | |
| 8b-b1 | marlin | 0.049 | 0.050 | 0.7 | x0.65 | |
| 8b-b4 | fp16 | 0.031 | 0.032 | 4.3 | x1.00 | |
| 8b-b4 | marlin | 0.049 | 0.050 | 2.7 | x0.64 | |
| 8b-b8 | fp16 | 0.032 | 0.032 | 8.5 | x1.00 | |
| 8b-b8 | marlin | 0.049 | 0.050 | 5.4 | x0.64 | |
| 70b-tp2-b1 | fp16 | 0.051 | 0.059 | 1.3 | x1.00 | |
| 70b-tp2-b1 | marlin | 0.050 | 0.051 | 1.4 | x1.03 | |
| 70b-tp2-b4 | fp16 | 0.049 | 0.050 | 5.4 | x1.00 | |
| 70b-tp2-b4 | marlin | 0.049 | 0.049 | 5.5 | x1.01 | |
| 70b-tp2-b8 | fp16 | 0.050 | 0.050 | 10.8 | x1.00 | |
| 70b-tp2-b8 | marlin | 0.048 | 0.049 | 11.1 | x1.02 | |

![roofline](./roofline.png)
