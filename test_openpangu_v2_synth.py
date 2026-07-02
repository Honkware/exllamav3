"""Synthetic correctness harness for the openpangu_v2 port (acceptance gate 2).

Generates a tiny random-weight openPangu checkpoint (every net-new path: mHC,
MoME, sinks, DSA + SWA + full attention, dense + MoE MLP, block-post norm),
loads it through BOTH the exllamav3 port and Huawei's pure-torch reference
(_pangu_torch_calib.py), and diffs them per sublayer site, per block, and on
final logits.

Usage (GPU box, reference file scp'd next to the model dir):
  python test_openpangu_v2_synth.py --gen --dir /tmp/pangu_synth
  python test_openpangu_v2_synth.py --run --dir /tmp/pangu_synth --ref /tmp/_pangu_torch_calib.py
"""
import argparse, json, os, sys
import torch

CFG = {
    "architectures": ["OpenPanguV2ForCausalLM"],
    "model_type": "openpangu_v2",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "q_lora_rank": 128,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 2,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "router_enable_expert_bias": True,
    "rms_norm_eps": 1e-5,
    "rope_theta": 6400000.0,
    "rope_interleave": False,
    "vocab_size": 512,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "sandwich_norm": True,
    "use_mhc": True,
    "mhc_num_stream": 4,
    "mhc_recur_norm": 20,
    "mhc_use_gamma": True,
    "use_mome": True,
    "router_sliding_window": 3,
    "param_sink_number": 4,
    "dsa_layers": [0, 3],
    "swa_layers": [1, 2],
    "sliding_window_list": [4, 8],
    "sliding_window": 4,
    "block_post_layernorm_idx": [0, 2],
    "index_topk": 64,
    "index_n_heads": 2,
    "index_head_dim": 64,
    "torch_dtype": "float16",
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def gen_checkpoint(model_dir):
    from safetensors.torch import save_file
    torch.manual_seed(7)
    os.makedirs(model_dir, exist_ok = True)
    H, N = CFG["hidden_size"], CFG["mhc_num_stream"]
    heads = CFG["num_attention_heads"]
    qkh = CFG["qk_nope_head_dim"] + CFG["qk_rope_head_dim"]
    t = {}

    def lin(key, out_f, in_f, std = 0.02):
        t[key] = (torch.randn(out_f, in_f) * std).half()

    def vec(key, n, mean = 1.0, std = 0.1):
        t[key] = (mean + torch.randn(n) * std).half()

    t["model.embed_tokens.weight"] = (torch.randn(CFG["vocab_size"], H) * 0.02).half()
    vec("model.norm.weight", H)
    lin("lm_head.weight", CFG["vocab_size"], H)
    lin("model.merge_mhc_module.phi.weight", N, N * H, 0.05)
    vec("model.merge_mhc_module.norm_gamma", N * H)
    t["model.merge_mhc_module.branch_alpha_pre"] = (torch.randn(1) * 0.5).half()
    t["model.merge_mhc_module.branch_beta_pre"] = (torch.randn(N) * 0.5).half()

    for i in range(CFG["num_hidden_layers"]):
        p = f"model.layers.{i}"
        for nk in ("input_layernorm", "post_attention_layernorm", "pre_mlp_layernorm", "post_mlp_layernorm"):
            vec(f"{p}.{nk}.weight", H)
        lin(f"{p}.self_attn.q_a_proj.weight", CFG["q_lora_rank"], H)
        vec(f"{p}.self_attn.q_a_layernorm.weight", CFG["q_lora_rank"])
        lin(f"{p}.self_attn.q_b_proj.weight", heads * qkh, CFG["q_lora_rank"])
        lin(f"{p}.self_attn.kv_a_proj_with_mqa.weight", CFG["kv_lora_rank"] + CFG["qk_rope_head_dim"], H)
        vec(f"{p}.self_attn.kv_a_layernorm.weight", CFG["kv_lora_rank"])
        lin(f"{p}.self_attn.kv_b_proj.weight", heads * (CFG["qk_nope_head_dim"] + CFG["v_head_dim"]), CFG["kv_lora_rank"])
        lin(f"{p}.self_attn.o_proj.weight", H, heads * CFG["v_head_dim"])
        t[f"{p}.self_attn.param_sink_compressed_kv"] = (torch.randn(CFG["param_sink_number"], CFG["kv_lora_rank"]) * 0.5).half()
        t[f"{p}.self_attn.param_sink_k_pe"] = (torch.randn(CFG["param_sink_number"], CFG["qk_rope_head_dim"]) * 0.5).half()
        for ck, d in (("qa_conv", CFG["q_lora_rank"]), ("compresskv_conv", CFG["kv_lora_rank"]), ("o_conv", heads * CFG["v_head_dim"])):
            t[f"{p}.self_attn.{ck}.weight"] = (torch.randn(d, 1, CFG["router_sliding_window"]) * 0.1).half()
        for m in ("attn_mhc_module", "mlp_mhc_module"):
            lin(f"{p}.{m}.phi.weight", N * (N + 2), N * H, 0.05)
            vec(f"{p}.{m}.norm_gamma", N * H)
            t[f"{p}.{m}.branch_alpha"] = (torch.randn(3) * 0.5).half()
            t[f"{p}.{m}.branch_beta"] = (torch.randn(N * (N + 2)) * 0.5).half()
        if i in CFG["block_post_layernorm_idx"]:
            vec(f"{p}.block_post_layernorm.weight", N * H)
        if i in CFG["dsa_layers"]:
            lin(f"{p}.self_attn.indexer.wq_b.weight", CFG["index_n_heads"] * CFG["index_head_dim"], CFG["q_lora_rank"])
            lin(f"{p}.self_attn.indexer.wk.weight", CFG["index_head_dim"], H)
            vec(f"{p}.self_attn.indexer.k_norm.weight", CFG["index_head_dim"])
            lin(f"{p}.self_attn.indexer.weights_proj.weight", CFG["index_n_heads"], H)
        if i < CFG["first_k_dense_replace"]:
            lin(f"{p}.mlp.gate_proj.weight", CFG["intermediate_size"], H)
            lin(f"{p}.mlp.up_proj.weight", CFG["intermediate_size"], H)
            lin(f"{p}.mlp.down_proj.weight", H, CFG["intermediate_size"])
        else:
            lin(f"{p}.mlp.gate.weight", CFG["n_routed_experts"], H)
            t[f"{p}.mlp.e_score_correction_bias"] = (torch.randn(CFG["n_routed_experts"]) * 0.01).half()
            for sk in ("gate_proj", "up_proj", "down_proj"):
                io = (H, CFG["moe_intermediate_size"]) if sk == "down_proj" else (CFG["moe_intermediate_size"], H)
                lin(f"{p}.mlp.shared_experts.{sk}.weight", *io)
            for e in range(CFG["n_routed_experts"]):
                for sk in ("gate_proj", "up_proj", "down_proj"):
                    io = (H, CFG["moe_intermediate_size"]) if sk == "down_proj" else (CFG["moe_intermediate_size"], H)
                    lin(f"{p}.mlp.experts.{e}.{sk}.weight", *io)

    save_file(t, os.path.join(model_dir, "model.safetensors"))
    index = {"metadata": {"total_size": sum(v.numel() * 2 for v in t.values())},
             "weight_map": {k: "model.safetensors" for k in t}}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(CFG, f, indent = 2)
    print(f"generated {len(t)} tensors -> {model_dir}")


def diff(name, a, b, results):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.shape != b.shape:
        results.append((name, float("inf"), 0.0, f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"))
        return
    mad = (a - b).abs().max().item()
    denom = b.abs().max().item() or 1.0
    cos = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
    results.append((name, mad, mad / denom, f"cos {cos:.6f}"))


@torch.inference_mode()
def run_test(model_dir, ref_path, device = "cuda:0", ref_dir = None):
    # ref_dir: read the torch reference from a different (unquantized) copy,
    # for round-trip testing of a converted model_dir
    sys.path.insert(0, os.path.dirname(os.path.abspath(ref_path)))
    ref_mod = __import__(os.path.basename(ref_path).replace(".py", ""))

    from exllamav3 import Config, Model
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load()

    ref = ref_mod.PanguTorchCalibModel(ref_dir or model_dir, full_layer = True)
    T = 24
    torch.manual_seed(11)
    ids = torch.randint(0, CFG["vocab_size"], (1, T))

    results = []
    from exllamav3.modules.arch_specific.pangu_v2 import PanguDecoderBlock, PanguMHC

    # reference forward, stepwise
    r_h = ref.model.embed_tokens(ids).to(device)
    my_h = model.modules[0].forward(ids, {}).to(device)
    diff("embed", my_h.half(), r_h, results)

    blocks = [m for m in model.modules if isinstance(m, PanguDecoderBlock)]
    n, H = CFG["mhc_num_stream"], CFG["hidden_size"]
    my_x = my_h
    for i, (blk, rlayer) in enumerate(zip(blocks, ref.model.layers)):
        rl = rlayer._ensure_loaded().to(device)

        # reference block, sub-stepped (mirrors _forward_full_layer)
        rx = r_h.reshape(-1, r_h.shape[-1]) if r_h.dim() == 3 and r_h.shape[-1] in (H, n * H) else r_h.reshape(-1, n, H)
        if rx.dim() == 2 and rx.shape[-1] == H:
            rx = rx.view(-1, 1, H).repeat(1, n, 1)
        elif rx.dim() == 2:
            rx = rx.view(-1, n, H)
        r_res = rx.clone()
        r_y, r_hpost, r_hres = rl.attn_mhc_module.mhc_pre(rx)
        r_y2 = rl.input_layernorm(r_y)
        r_attn = rl.self_attn(r_y2)
        r_y3 = rl.post_attention_layernorm(r_attn)
        r_hres = rl.attn_mhc_module.mhc_sinkhorn(r_hres)
        rx = rl.attn_mhc_module.mhc_post(r_y3, r_hpost, r_res, r_hres).view(-1, n, H)
        r_res = rx.clone()
        m_y, m_hpost, m_hres = rl.mlp_mhc_module.mhc_pre(rx)
        m_y2 = rl.pre_mlp_layernorm(m_y)
        r_mlp = rl.mlp(m_y2)
        m_y3 = rl.post_mlp_layernorm(r_mlp)
        m_hres = rl.mlp_mhc_module.mhc_sinkhorn(m_hres)
        rx = rl.mlp_mhc_module.mhc_post(m_y3, m_hpost, r_res, m_hres)
        if rl.has_block_post_layernorm:
            rx = rl.block_post_layernorm(rx.view(-1, n * H)).view(-1, n, H)
        r_block_out = rx

        # my block, sub-stepped identically
        x = my_x.half().view(1 * T, -1)
        x = x.view(-1, 1, H).repeat(1, n, 1) if x.shape[-1] == H else x.view(-1, n, H)
        res = x
        y, hpost, hres = blk.attn_mhc.mhc_pre(x)
        diff(f"L{i}.attn.mhc_pre", y, r_y, results)
        y = blk.input_norm.forward(y, {}, out_dtype = torch.half)
        diff(f"L{i}.attn.input_norm", y, r_y2, results)
        y = blk.attn.forward(y, {})
        diff(f"L{i}.attn.out", y, r_attn, results)
        y = blk.post_attn_norm.forward(y, {}, out_dtype = torch.half)
        diff(f"L{i}.attn.post_norm", y, r_y3, results)
        hres = blk.attn_mhc.mhc_sinkhorn(hres)
        x = blk.attn_mhc.mhc_post(y, hpost, res, hres)
        diff(f"L{i}.attn.mhc_post", x, r_res, results)
        res = x
        y, hpost, hres = blk.mlp_mhc.mhc_pre(x)
        diff(f"L{i}.mlp.mhc_pre", y, m_y, results)
        y = blk.pre_mlp_norm.forward(y, {}, out_dtype = torch.half)
        diff(f"L{i}.mlp.pre_norm", y, m_y2, results)
        y = blk.mlp.forward(y.view(1, -1, H), {})
        diff(f"L{i}.mlp.out", y.half(), r_mlp, results)
        y = blk.post_mlp_norm.forward(y.half().reshape(-1, H), {}, out_dtype = torch.half)
        diff(f"L{i}.mlp.post_norm", y, m_y3, results)
        hres = blk.mlp_mhc.mhc_sinkhorn(hres)
        x = blk.mlp_mhc.mhc_post(y, hpost, res, hres)
        if blk.block_post_norm is not None:
            x = blk.block_post_norm.forward(x.reshape(-1, n * H), {}, out_dtype = torch.half)
        diff(f"L{i}.block_out", x.reshape(-1), r_block_out.reshape(-1), results)

        # advance both via the REAL block forwards (catches harness drift)
        my_x = blk.forward(my_x, {})
        r_h = rl(r_h)
        diff(f"L{i}.fullfwd", my_x, r_h.reshape(1, T, -1), results)

    # tail: merge -> norm -> head
    merge = next(m for m in model.modules if isinstance(m, PanguMHC) and m.pre_only)
    my_t = merge.forward(my_x, {})
    r_flat = r_h.reshape(-1, n, H)
    r_t, _, _ = ref.model.merge_mhc_module.to(device).mhc_pre(r_flat)
    diff("merge", my_t, r_t, results)
    my_t = model.modules[-2].forward(my_t, {}, out_dtype = torch.half)
    r_t = ref.model.norm.to(device)(r_t)
    diff("final_norm", my_t, r_t, results)
    my_logits = model.modules[-1].forward(my_t, {})
    lm_w = ref.store.get_tensor("lm_head.weight").to(device)
    r_logits = torch.nn.functional.linear(r_t.half(), lm_w)
    diff("logits", my_logits, r_logits, results)

    print(f"\n{'site':26s} {'max_abs':>10s} {'rel':>10s}")
    worst = 0.0
    for name, mad, rel, extra in results:
        flag = " <-- FAIL" if rel > 3e-2 or mad != mad else ""
        print(f"{name:26s} {mad:10.6f} {rel:10.6f}  {extra}{flag}")
        worst = max(worst, rel)
    print(f"\nWORST rel diff: {worst:.6f}  ({'PASS' if worst <= 3e-2 else 'FAIL'} @ 3e-2)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default = "/tmp/pangu_synth")
    ap.add_argument("--ref", default = "/tmp/_pangu_torch_calib.py")
    ap.add_argument("--ref-dir", default = None)
    ap.add_argument("--gen", action = "store_true")
    ap.add_argument("--run", action = "store_true")
    args = ap.parse_args()
    if args.gen:
        gen_checkpoint(args.dir)
    if args.run:
        run_test(args.dir, args.ref, ref_dir = args.ref_dir)
