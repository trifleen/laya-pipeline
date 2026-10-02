#!/usr/bin/env bash
# Build expert-trace against the llama.cpp checkout and build in LLAMA_CPP (default
# ~/Work/moe-offload/llama.cpp). The binary lands next to this script.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
L=${LLAMA_CPP:-$HOME/Work/moe-offload/llama.cpp}
g++ -O2 -std=c++17 "$here/expert_trace.cpp" -o "$here/expert-trace" \
  -I "$L/include" -I "$L/ggml/include" -I "$L/vendor" \
  -L "$L/build/bin" -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin"
echo "built $here/expert-trace"
