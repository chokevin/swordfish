# Autoresearch run `20260420T020830Z`

- **source SHA:** `eb0f6e3`
- **GPU:** NVIDIA A100-SXM4-80GB (cc 8.0, 79.3 GB)
- **CUDA / torch / triton:** 12.4 / 2.4.0a0+07cecf4168.nv24.05 / 3.0.0
- **shapes:** `voice`  **impls:** `fp16,marlin`  **repeats:** 5
- **marlin SHA:** `1f25790bdd49fba53106164a24666dade68d7c90`

## Results

| shape | impl | ms_mean | ms_p95 | TFLOPS | speedup vs fp16 | error |
|---|---|---|---|---|---|---|
| 8b-b1 | fp16 | 0.031 | 0.032 | 1.1 | x1.00 | |
| 8b-b1 | marlin | 0.050 | 0.055 | 0.7 | x0.62 | |
| 8b-b4 | fp16 | 0.031 | 0.031 | 4.3 | x1.00 | |
| 8b-b4 | marlin | 0.049 | 0.049 | 2.7 | x0.63 | |
| 8b-b8 | fp16 | 0.031 | 0.032 | 8.6 | x1.00 | |
| 8b-b8 | marlin | 0.049 | 0.050 | 5.5 | x0.63 | |
| 70b-tp2-b1 | fp16 | 0.050 | 0.055 | 1.3 | x1.00 | |
| 70b-tp2-b1 | marlin | 0.051 | 0.052 | 1.3 | x0.99 | |
| 70b-tp2-b4 | fp16 | 0.049 | 0.049 | 5.5 | x1.00 | |
| 70b-tp2-b4 | marlin | 0.049 | 0.050 | 5.4 | x1.00 | |
| 70b-tp2-b8 | fp16 | 0.049 | 0.050 | 10.9 | x1.00 | |
| 70b-tp2-b8 | marlin | 0.049 | 0.050 | 10.9 | x1.00 | |

![roofline](./roofline.png)
