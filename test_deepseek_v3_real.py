"""Gate 9 (real weights) for the MLA substrate: Moonlight-16B-A3B
(DeepseekV3ForCausalLM, q_lora_rank=None, 27 layers, 64-expert MoE).

--audit    key coverage vs the checkpoint index
--forward  logits: exllamav3 substrate (GPU, fp16) vs HF transformers (CPU, bf16)

Convert + round-trip run via convert.py / examples separately.
"""
import argparse, json, os
import torch

MD = "/root/models/Moonlight-16B-A3B"


def audit(md):
    from exllamav3 import Config, Model
    config = Config.from_directory(md)
    model = Model.from_config(config)
    print(f"arch: {type(model).__name__} | layers: {config.num_hidden_layers} | modules: {len(model.modules)}")
    exp = set()
    exp.update(["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"])
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        exp.add(f"{p}.input_layernorm.weight")
        exp.add(f"{p}.post_attention_layernorm.weight")
        if config.q_lora_rank:
            exp.update([f"{p}.self_attn.q_a_proj.weight", f"{p}.self_attn.q_a_layernorm.weight",
                        f"{p}.self_attn.q_b_proj.weight"])
        else:
            exp.add(f"{p}.self_attn.q_proj.weight")
        exp.update([f"{p}.self_attn.kv_a_proj_with_mqa.weight", f"{p}.self_attn.kv_a_layernorm.weight",
                    f"{p}.self_attn.kv_b_proj.weight", f"{p}.self_attn.o_proj.weight"])
        if i < config.first_k_dense_replace:
            for s in ("gate_proj", "up_proj", "down_proj"):
                exp.add(f"{p}.mlp.{s}.weight")
        else:
            exp.add(f"{p}.mlp.gate.weight")
            exp.add(f"{p}.mlp.gate.e_score_correction_bias")
            for s in ("gate_proj", "up_proj", "down_proj"):
                exp.add(f"{p}.mlp.shared_experts.{s}.weight")
            for e in range(config.num_experts):
                for s in ("gate_proj", "up_proj", "down_proj"):
                    exp.add(f"{p}.mlp.experts.{e}.{s}.weight")
    with open(os.path.join(md, "model.safetensors.index.json")) as f:
        wm = set(json.load(f)["weight_map"].keys())
    benign = {k for k in wm if k.endswith("rotary_emb.inv_freq")}  # legacy non-learned buffers
    unexpected = wm - exp - benign
    missing = exp - wm
    print(f"checkpoint tensors: {len(wm)} | mapped: {len(wm & exp)} | benign inv_freq: {len(benign)} | UNEXPECTED: {len(unexpected)} | missing: {len(missing)}")
    for k in sorted(unexpected)[:10]: print("  unexpected:", k)
    for k in sorted(missing)[:10]: print("  missing:", k)
    print("AUDIT:", "PASS" if not unexpected and not missing else "FAIL")


@torch.inference_mode()
def forward_match(md):
    T = 24
    torch.manual_seed(31)
    ids = torch.randint(0, 163840, (1, T))

    from exllamav3 import Config, Model
    config = Config.from_directory(md)
    model = Model.from_config(config)
    model.load()
    x = model.modules[0].forward(ids, {}).to("cuda:0")
    for m in model.modules[1:]:
        if x.dtype != torch.half:
            x = x.half()
        x = m.forward(x, {})
    my_logits = x.float().cpu()
    model.unload()
    del model
    torch.cuda.empty_cache()
    print("mine done:", tuple(my_logits.shape))

    # HF DS3 MoE needs aten::_grouped_mm (CUDA-only); mine is unloaded, so the
    # GPU is free for the reference. Run the reference in BOTH fp16 and bf16:
    # their spread is the model's own dtype-noise floor, which calibrates what
    # "matches" means for a 27-layer forward on real weights.
    from transformers import AutoModelForCausalLM
    r = {}
    for dt in (torch.float16, torch.bfloat16):
        ref = AutoModelForCausalLM.from_pretrained(md, dtype = dt, attn_implementation = "eager").to("cuda:0")
        ref.eval()
        r[dt] = ref(input_ids = ids.to("cuda:0"), use_cache = False).logits.float().cpu()
        del ref
        torch.cuda.empty_cache()
        print(f"ref {dt} done")

    def stats(tag, a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        mad = (a - b).abs().max().item()
        rel = mad / (b.abs().max().item() or 1.0)
        cos = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
        tm = (a.view(1, T, -1)[0].argmax(-1) == b.view(1, T, -1)[0].argmax(-1)).float().mean().item()
        print(f"{tag:24s} max_abs {mad:.4f} rel {rel:.4f} cos {cos:.6f} argmax {tm:.1%}")
        return rel, cos, tm

    floor_rel, _, floor_tm = stats("HF fp16 vs HF bf16", r[torch.float16], r[torch.bfloat16])
    rel16, cos16, tm16 = stats("mine vs HF fp16", my_logits, r[torch.float16])
    stats("mine vs HF bf16", my_logits, r[torch.bfloat16])
    ok = rel16 <= max(3e-2, 1.5 * floor_rel) and cos16 > 0.999 and tm16 >= floor_tm - 0.05
    print("FORWARD:", "PASS (within reference dtype-noise floor)" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action = "store_true")
    ap.add_argument("--forward", action = "store_true")
    args = ap.parse_args()
    if args.audit:
        audit(MD)
    if args.forward:
        forward_match(MD)
