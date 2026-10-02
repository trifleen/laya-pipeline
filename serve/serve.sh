#!/usr/bin/env bash
# One llama-server in router mode (OpenAI-compatible API at http://localhost:8080/v1).
#   serve/serve.sh                              Qwen MoE + Nemotron 4B + embedder (models.ini.in)
#   PRESET=qwen-only N_CPU_MOE=34 serve/serve.sh  Qwen MoE + embedder, 6 expert layers on the GPU
# LLAMA_CPP: a llama.cpp checkout with build/bin/llama-server. MODELS: the folder with the GGUFs
# (see serve/README.md). The presets are rendered into serve/.generated/ with those paths.
here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
LLAMA_CPP=${LLAMA_CPP:-$HOME/Work/moe-offload/llama.cpp}
MODELS=${MODELS:-$HOME/Work/moe-offload/models}
mkdir -p "$here/.generated"
if [[ ${PRESET:-} == qwen-only ]]; then
  ini="$here/.generated/models-qwen-only.ini"
  sed -e "s|__MODELS__|$MODELS|g" -e "s/__N_CPU_MOE__/${N_CPU_MOE:-40}/" "$here/models-qwen-only.ini.in" > "$ini"
else
  ini="$here/.generated/models.ini"
  sed "s|__MODELS__|$MODELS|g" "$here/models.ini.in" > "$ini"
fi
exec "$LLAMA_CPP/build/bin/llama-server" --models-preset "$ini" \
  --models-max 3 --host 127.0.0.1 --port "${PORT:-8080}" "$@"
