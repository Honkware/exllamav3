"""Teacher-forced logit parity for the pangu paged path: the same tokens
through (a) no-cache forward, (b) one-shot cached forward, (c) token-by-token
cached decode. Exercises the conv ring state, sink logsumexp merge and
DSA-dense-over-cache against the Phase-1 full-sequence math."""
import os, torch
from exllamav3 import Config, Model, Cache, Tokenizer

MD = os.environ.get("PANGU_MD", "/root/models/openPangu-exl3")
T = 96

config = Config.from_directory(MD)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096)
model.load()
tok = Tokenizer.from_config(config)

ids = tok.encode("The chemical composition of water is H2O, meaning each molecule contains "
                 "two hydrogen atoms and one oxygen atom. Water covers about seventy percent "
                 "of the surface of the Earth and is essential to all known forms of life.",
                 add_bos = True)[:, :T]
T = ids.shape[-1]
n_pages = (4096 + 255) // 256

@torch.inference_mode()
def run_nocache():
    x = model.modules[0].forward(ids, {}).to("cuda:0")
    for m in model.modules[1:]:
        if x.dtype != torch.half: x = x.half()
        x = m.forward(x, {})
    return x.float().cpu()

def paged_params(seqlen):
    return {
        "attn_mode": "flash_attn",
        "cache": cache,
        "block_table": torch.arange(n_pages, dtype = torch.int32).unsqueeze(0),
        "cache_seqlens": torch.tensor([seqlen], dtype = torch.int32),
    }

@torch.inference_mode()
def run_oneshot():
    return model.forward(input_ids = ids, params = paged_params(0)).float().cpu()

@torch.inference_mode()
def run_stepwise():
    outs = []
    for i in range(T):
        logits = model.forward(input_ids = ids[:, i:i+1], params = paged_params(i))
        outs.append(logits.float().cpu())
    return torch.cat(outs, dim = 1)

a = run_nocache()
@torch.inference_mode()
def reset_cache():
    for layer in cache.layers.values():
        if layer.k is not None: layer.k.zero_()
        if layer.v is not None: layer.v.zero_()
reset_cache()
b = run_oneshot()
reset_cache()
c = run_stepwise()

def cmp(tag, x, y):
    mad = (x - y).abs().max().item()
    rel = mad / (y.abs().max().item() or 1.0)
    cos = torch.nn.functional.cosine_similarity(x.reshape(-1), y.reshape(-1), dim = 0).item()
    am = (x[0].argmax(-1) == y[0].argmax(-1)).float().mean().item()
    print(f"{tag:28s} rel {rel:.5f} cos {cos:.6f} argmax {am:.1%}")
    return rel, am

r1, a1 = cmp("oneshot-cached vs nocache", b, a)
r2, a2 = cmp("stepwise-cached vs nocache", c, a)
r3, a3 = cmp("stepwise vs oneshot", c, b)
ok = a1 > 0.95 and a2 > 0.95 and r1 < 5e-2 and r2 < 5e-2
print("PANGU PAGED PARITY:", "PASS" if ok else "FAIL")
