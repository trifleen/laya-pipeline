# The model server

One `llama-server` in router mode serves all three models the pipeline uses. Measurements behind
every setting are in [`bench/results/phase0/SUMMARY.md`](../bench/results/phase0/SUMMARY.md) and
the experiment log, [`docs/experiments/2026-10-02-moe-offload.md`](../docs/experiments/2026-10-02-moe-offload.md).

## Setup

1. Build llama.cpp with CUDA (tested at commit `4ebdf2c`):

   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j --target llama-server llama-bench
   ```

2. Download the models into one folder (~25 GB):

   ```bash
   base=https://huggingface.co
   curl -L -C - -o Qwen3.6-35B-A3B-UD-Q4_K_M.gguf $base/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
   curl -L -C - -o NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf $base/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf
   ```

   plus `nomic-embed-text-v1.5.Q4_K_M.gguf` (bundled with LM Studio, or from `nomic-ai/nomic-embed-text-v1.5-GGUF`).

3. Point the script at both and start it (`./laya up` does this for you when nothing answers on port 8080):

   ```bash
   LLAMA_CPP=~/src/llama.cpp MODELS=~/models serve/serve.sh
   ```

   Set the same `LLAMA_CPP` and `MODELS` in the environment of `./laya up`.

## Presets

| | VRAM | When to use |
|---|---|---|
| `serve/serve.sh` (default) | ~6.5 GB + Laya ~1 GB | Fastest pipeline: easy requests go to the 4B model |
| `PRESET=qwen-only N_CPU_MOE=34 serve/serve.sh`, with `SMALL_MODEL=qwen3.6-35b-a3b` for Laya | ~6.2 GB + Laya | Every answer from the 35B; ~20% slower overall, slightly better answers |

Tuned for an 8 GB GPU with 30 GB of RAM: the 35B model's experts (~19 GB) are read from RAM
through the page cache, so keep ~22 GB of RAM free.
