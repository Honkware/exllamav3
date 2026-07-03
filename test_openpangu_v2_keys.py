"""Key audit for the openpangu_v2 port (acceptance gate 1).

Downloads just config.json + model.safetensors.index.json (+ tokenizer) from the
real repo, builds the Model (no weight load), enumerates every tensor key the
port expects, and classifies the checkpoint's weight_map:
  mapped              -- consumed by a module
  skipped_mtp         -- layers >= num_hidden_layers (MTP, dropped in Phase 1)
  UNEXPECTED          -- anything else (must be zero)
Also reports expected-but-missing keys (must be zero).

Usage: python test_openpangu_v2_keys.py [--dir DIR] [--fetch]
"""
import argparse, json, os
import torch

REPO = "openpangu/openPangu-2.0-Flash"


def fetch_meta(model_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(model_dir, exist_ok = True)
    for f in ("config.json", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        try:
            hf_hub_download(REPO, f, local_dir = model_dir)
        except Exception as e:
            print(f"  (skip {f}: {e})")
    print("meta fetched ->", model_dir)


def expected_keys(config):
    keys = set()
    H = config.hidden_size
    keys.add("model.embed_tokens.weight")
    keys.add("model.norm.weight")
    keys.add("lm_head.weight")
    for s in ("phi.weight", "norm_gamma", "branch_alpha_pre", "branch_beta_pre"):
        keys.add(f"model.merge_mhc_module.{s}")
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        for nk in ("input_layernorm", "post_attention_layernorm", "pre_mlp_layernorm", "post_mlp_layernorm"):
            keys.add(f"{p}.{nk}.weight")
        for sk in ("q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
                   "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
                   "o_proj.weight", "param_sink_compressed_kv", "param_sink_k_pe",
                   "qa_conv.weight", "compresskv_conv.weight", "o_conv.weight"):
            keys.add(f"{p}.self_attn.{sk}")
        for m in ("attn_mhc_module", "mlp_mhc_module"):
            for s in ("phi.weight", "norm_gamma", "branch_alpha", "branch_beta"):
                keys.add(f"{p}.{m}.{s}")
        if i in config.block_post_layernorm_idx:
            keys.add(f"{p}.block_post_layernorm.weight")
        if i in config.dsa_layers:
            for s in ("wq_b.weight", "wk.weight", "k_norm.weight", "weights_proj.weight"):
                keys.add(f"{p}.self_attn.indexer.{s}")
        if i < config.first_k_dense_replace:
            for s in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{p}.mlp.{s}.weight")
        else:
            keys.add(f"{p}.mlp.gate.weight")
            keys.add(f"{p}.mlp.e_score_correction_bias")
            for s in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{p}.mlp.shared_experts.{s}.weight")
            for e in range(config.num_experts):
                for s in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{p}.mlp.experts.{e}.{s}.weight")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default = "/root/models/openPangu-2.0-Flash-meta")
    ap.add_argument("--fetch", action = "store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch_meta(args.dir)

    from exllamav3 import Config, Model
    config = Config.from_directory(args.dir)
    model = Model.from_config(config)
    print(f"arch: {type(model).__name__} | layers: {config.num_hidden_layers} | modules: {len(model.modules)}")

    exp = expected_keys(config)
    with open(os.path.join(args.dir, "model.safetensors.index.json")) as f:
        wm = set(json.load(f)["weight_map"].keys())

    import re
    mtp_re = re.compile(r"^model\.layers\.(\d+)\.")
    def is_mtp(k):
        m = mtp_re.match(k)
        return m and int(m.group(1)) >= config.num_hidden_layers

    mapped = wm & exp
    skipped_mtp = {k for k in wm - exp if is_mtp(k)}
    unexpected = wm - exp - skipped_mtp
    missing = exp - wm

    print(f"checkpoint tensors: {len(wm)}")
    print(f"  mapped:      {len(mapped)}")
    print(f"  skipped_mtp: {len(skipped_mtp)} (layers >= {config.num_hidden_layers}, dropped by design)")
    print(f"  UNEXPECTED:  {len(unexpected)}")
    for k in sorted(unexpected)[:20]: print("    ", k)
    print(f"expected-but-missing: {len(missing)}")
    for k in sorted(missing)[:20]: print("    ", k)
    ok = not unexpected and not missing
    print("GATE1:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
