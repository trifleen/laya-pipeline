# Phase 0: stock llama.cpp expert offload

Qwen3.6-35B-A3B UD-Q4_K_M (22.1 GB, sha256 ac0e2c11…), llama.cpp 4ebdf2c (CUDA, sm_86),
RTX 3070 Laptop 8 GB, Ryzen 9 5900HX 8 threads, 30 GB DDR4. `-ngl 99 -fa on`, 2026-10-02.

`--n-cpu-moe N` = the first N of the model's layers keep their experts in RAM (computed on the CPU).

| n-cpu-moe | prompt 512 | prompt 4096 | generate (empty ctx) | generate @ 8k ctx | generate @ 32k ctx |
|---|---|---|---|---|---|
| 40 (all experts in RAM) | 237 tok/s | 226 tok/s | 42.3 tok/s | | |
| 36 | 256 | 244 | 45.5 | | 40.3 |
| 34 | | | | | 40.1 |
| 32 | 281 | 268 | 48.2 | | 42.8 |
| 30 | 293 | 280 | 49.8 | 44.0 | does not fit |
| ≤ 29 | does not fit in 8 GB VRAM | | | | |

Raw: `sweep-20261002-195614.jsonl`, `depth-ncmoe30.jsonl`, `depth32k.txt`.

Takeaways
- Generation is fast: 42–50 tok/s, because only ~3B parameters (~0.6 GB of expert weights) are read per token.
- Each 2 layers of experts moved to the GPU adds ~1.5 tok/s; VRAM runs out at 30.
- VRAM is shared between experts and KV cache: 30 suits ≤ 8k contexts, 32 is the best setting for 32k.
- Prompt reading (225–295 tok/s) is the weak spot: a 2k-token prompt takes ~7 s before the first token.
