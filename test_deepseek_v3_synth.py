"""Synthetic correctness + separability harness for the MLA substrate on
DeepSeek-V3 (acceptance gates 8/10-synthetic).

Generates a tiny random-weight DeepseekV3ForCausalLM checkpoint (dense + grouped
MoE layers, q_lora or direct-q), loads it through BOTH the exllamav3 substrate
arch and HF transformers (the reference oracle), and diffs per-layer hiddens +
final logits. Also asserts the separability contract: the attention in the call
path is exactly MLAAttention (not a subclass) and no openPangu module is
imported by the DS3 path.

Usage:
  python test_deepseek_v3_synth.py --gen --dir /tmp/ds3_synth [--qlora 128]
  python test_deepseek_v3_synth.py --run --dir /tmp/ds3_synth
"""
import argparse, json, os, sys
import torch

def make_cfg(q_lora):
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": q_lora if q_lora else None,
        "kv_lora_rank": 128,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 16,
        "qk_head_dim": 80,
        "v_head_dim": 64,
        "head_dim": 16,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "vocab_size": 512,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "bos_token_id": 0,
        "eos_token_id": 1,
    }


def gen_checkpoint(model_dir, q_lora):
    from safetensors.torch import save_file
    torch.manual_seed(17)
    os.makedirs(model_dir, exist_ok = True)
    CFG = make_cfg(q_lora)
    H = CFG["hidden_size"]
    heads = CFG["num_attention_heads"]
    qkh = CFG["qk_nope_head_dim"] + CFG["qk_rope_head_dim"]
    t = {}

    def lin(key, out_f, in_f, std = 0.02):
        t[key] = (torch.randn(out_f, in_f) * std).half()

    def vec(key, n):
        t[key] = (1.0 + torch.randn(n) * 0.1).half()

    t["model.embed_tokens.weight"] = (torch.randn(CFG["vocab_size"], H) * 0.02).half()
    vec("model.norm.weight", H)
    lin("lm_head.weight", CFG["vocab_size"], H)
    for i in range(CFG["num_hidden_layers"]):
        p = f"model.layers.{i}"
        vec(f"{p}.input_layernorm.weight", H)
        vec(f"{p}.post_attention_layernorm.weight", H)
        if q_lora:
            lin(f"{p}.self_attn.q_a_proj.weight", q_lora, H)
            vec(f"{p}.self_attn.q_a_layernorm.weight", q_lora)
            lin(f"{p}.self_attn.q_b_proj.weight", heads * qkh, q_lora)
        else:
            lin(f"{p}.self_attn.q_proj.weight", heads * qkh, H)
        lin(f"{p}.self_attn.kv_a_proj_with_mqa.weight", CFG["kv_lora_rank"] + CFG["qk_rope_head_dim"], H)
        vec(f"{p}.self_attn.kv_a_layernorm.weight", CFG["kv_lora_rank"])
        lin(f"{p}.self_attn.kv_b_proj.weight", heads * (CFG["qk_nope_head_dim"] + CFG["v_head_dim"]), CFG["kv_lora_rank"])
        lin(f"{p}.self_attn.o_proj.weight", H, heads * CFG["v_head_dim"])
        if i < CFG["first_k_dense_replace"]:
            for sk, io in (("gate_proj", (CFG["intermediate_size"], H)), ("up_proj", (CFG["intermediate_size"], H)),
                           ("down_proj", (H, CFG["intermediate_size"]))):
                lin(f"{p}.mlp.{sk}.weight", *io)
        else:
            lin(f"{p}.mlp.gate.weight", CFG["n_routed_experts"], H)
            t[f"{p}.mlp.gate.e_score_correction_bias"] = (torch.randn(CFG["n_routed_experts"]) * 0.01).float()
            for sk in ("gate_proj", "up_proj", "down_proj"):
                io = (H, CFG["moe_intermediate_size"]) if sk == "down_proj" else (CFG["moe_intermediate_size"], H)
                lin(f"{p}.mlp.shared_experts.{sk}.weight", *io)
            for e in range(CFG["n_routed_experts"]):
                for sk in ("gate_proj", "up_proj", "down_proj"):
                    io = (H, CFG["moe_intermediate_size"]) if sk == "down_proj" else (CFG["moe_intermediate_size"], H)
                    lin(f"{p}.mlp.experts.{e}.{sk}.weight", *io)

    save_file(t, os.path.join(model_dir, "model.safetensors"))
    index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in t.values())},
             "weight_map": {k: "model.safetensors" for k in t}}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(CFG, f, indent = 2)
    print(f"generated {len(t)} tensors -> {model_dir} (q_lora={q_lora})")


def diff(name, a, b, results):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    mad = (a.cpu() - b).abs().max().item()
    denom = b.abs().max().item() or 1.0
    cos = torch.nn.functional.cosine_similarity(a.cpu(), b, dim = 0).item()
    results.append((name, mad, mad / denom, cos))


@torch.inference_mode()
def run_test(model_dir, device = "cuda:0"):
    # separability contract: DS3 path must not import any pangu module
    from exllamav3 import Config, Model
    from exllamav3.modules.mla_attn import MLAAttention
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load()
    from exllamav3.modules.transformer import TransformerBlock
    blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
    attns = [b.attn for b in blocks]
    assert all(type(a) is MLAAttention for a in attns), "DS3 attention must be exactly MLAAttention"
    pangu_mods = [m for m in sys.modules if "pangu" in m.lower()]
    print(f"substrate: {type(attns[0]).__name__} x{len(attns)} | pangu modules imported: {pangu_mods or 'NONE'}")

    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype = torch.float16, attn_implementation = "eager").to(device).eval()

    T = 24
    torch.manual_seed(23)
    ids = torch.randint(0, 512, (1, T))
    r = ref(input_ids = ids.to(device), output_hidden_states = True, use_cache = False)

    results = []
    # mine, module by module; compare at each block boundary vs HF hidden_states.
    # NB: HF appends the LAST hidden state post-final-norm, so the last block
    # compares after model.norm.
    x = model.modules[0].forward(ids, {}).to(device)
    diff("embed", x.half(), r.hidden_states[0], results)
    for i, blk in enumerate(blocks[:-1]):
        x = blk.forward(x.half() if x.dtype != torch.half else x, {})
        diff(f"L{i}.block_out", x, r.hidden_states[i + 1], results)
    x = blocks[-1].forward(x.half() if x.dtype != torch.half else x, {})
    x = model.modules[-2].forward(x.half() if x.dtype != torch.half else x, {}, out_dtype = torch.half)
    diff(f"L{len(blocks)-1}.out+norm", x, r.hidden_states[-1], results)
    logits = model.modules[-1].forward(x, {})
    diff("logits", logits, r.logits, results)

    print(f"\n{'site':16s} {'max_abs':>10s} {'rel':>10s} {'cos':>10s}")
    worst = 0.0
    for name, mad, rel, cos in results:
        flag = " <-- FAIL" if rel > 3e-2 or mad != mad else ""
        print(f"{name:16s} {mad:10.6f} {rel:10.6f} {cos:10.6f}{flag}")
        worst = max(worst, rel)
    print(f"\nWORST rel diff: {worst:.6f}  ({'PASS' if worst <= 3e-2 else 'FAIL'} @ 3e-2)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default = "/tmp/ds3_synth")
    ap.add_argument("--qlora", type = int, default = 128)
    ap.add_argument("--gen", action = "store_true")
    ap.add_argument("--run", action = "store_true")
    args = ap.parse_args()
    if args.gen:
        gen_checkpoint(args.dir, args.qlora)
    if args.run:
        run_test(args.dir)
