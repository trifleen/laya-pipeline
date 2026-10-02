// expert-trace: record which experts a llama.cpp MoE model routes every token to.
//
//   expert-trace MODEL.gguf PROMPTS.jsonl OUT.bin [n_generate=192]
//
// PROMPTS.jsonl has one {"text": "<already chat-formatted prompt>"} per line. For each prompt
// the context is cleared, the prompt is read in 2,048-token batches ("prefill") and then up to
// n_generate tokens are generated greedily ("decode"). Every time a layer's router picks its
// experts (the graph tensor "ffn_moe_topk-<layer>", int32 [n_expert_used, n_tokens]), one
// record is appended to OUT.bin:
//
//   u16 prompt index, u8 phase (0 prefill, 1 decode), u8 layer, u16 n_tokens, u8 n_used,
//   then n_tokens * n_used u8 expert ids (fine for <= 256 experts)
//
// Expert placement doesn't change routing, so all expert tensors simply stay in RAM.

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <nlohmann/json.hpp>

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct trace_state {
    FILE * out = nullptr;
    uint16_t prompt = 0;
    uint8_t phase = 0;
    std::vector<int32_t> buf;
    uint64_t records = 0;
};

static bool on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char prefix[] = "ffn_moe_topk-";
    const bool want = strncmp(t->name, prefix, sizeof(prefix) - 1) == 0;
    if (ask) {
        return want;  // only these tensors are handed back with their data
    }
    if (!want || t->type != GGML_TYPE_I32) {
        return true;
    }
    auto * st = (trace_state *) user_data;
    const int n_used = (int) t->ne[0];
    const int n_tok = (int) t->ne[1];
    st->buf.resize((size_t) n_used * n_tok);
    // The tensor may be a view with strides; copy row by row.
    for (int i = 0; i < n_tok; ++i) {
        ggml_backend_tensor_get(t, st->buf.data() + (size_t) i * n_used, i * t->nb[1], n_used * sizeof(int32_t));
    }
    const uint8_t layer = (uint8_t) atoi(t->name + sizeof(prefix) - 1);
    const uint16_t nt = (uint16_t) n_tok;
    const uint8_t nu = (uint8_t) n_used;
    fwrite(&st->prompt, 2, 1, st->out);
    fwrite(&st->phase, 1, 1, st->out);
    fwrite(&layer, 1, 1, st->out);
    fwrite(&nt, 2, 1, st->out);
    fwrite(&nu, 1, 1, st->out);
    for (int32_t e : st->buf) {
        const uint8_t b = (uint8_t) e;
        fwrite(&b, 1, 1, st->out);
    }
    st->records++;
    return true;
}

static void set_tokens(llama_batch_ext * batch, const llama_token * tokens, int32_t n, llama_pos pos0) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n - 1, true);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf PROMPTS.jsonl OUT.bin [n_generate]\n", argv[0]);
        return 1;
    }
    const int n_generate = argc > 4 ? atoi(argv[4]) : 192;
    const int n_batch = 2048;

    std::vector<std::string> prompts;
    {
        std::ifstream in(argv[2]);
        for (std::string line; std::getline(in, line);) {
            if (!line.empty()) {
                prompts.push_back(nlohmann::json::parse(line).at("text").get<std::string>());
            }
        }
    }

    trace_state st;
    st.out = fopen(argv[3], "wb");
    if (!st.out) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    // Every expert tensor in RAM (like --n-cpu-moe 40); everything else on the GPU.
    llama_model_tensor_buft_override overrides[] = {
        {"\\.ffn_(up|down|gate|gate_up)_exps", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    mp.tensor_buft_overrides = overrides;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16384;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    for (size_t p = 0; p < prompts.size(); ++p) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);
        st.prompt = (uint16_t) p;

        const std::string & text = prompts[p];
        const int n_prompt = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> toks(n_prompt);
        llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, true);

        st.phase = 0;
        int pos = 0;
        for (int i = 0; i < n_prompt; i += n_batch) {
            const int n = std::min(n_batch, n_prompt - i);
            set_tokens(batch, toks.data() + i, n, pos);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "prompt %zu: decode failed\n", p);
                return 1;
            }
            pos += n;
        }

        st.phase = 1;
        int generated = 0;
        for (; generated < n_generate; ++generated) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            set_tokens(batch, &tok, 1, pos++);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "prompt %zu: decode failed\n", p);
                return 1;
            }
        }
        fprintf(stderr, "prompt %3zu/%zu: %5d prompt tokens, %3d generated, %llu records\n", p + 1, prompts.size(),
                n_prompt, generated, (unsigned long long) st.records);
    }

    fclose(st.out);
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
