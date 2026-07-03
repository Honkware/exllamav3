"""CUDA-graph capture of the pangu decode step: static device-resident params,
warmup for every lazy build, capture one trunk forward, replay-timed.
Measures the launch-overhead-free ceiling. EXLLAMA_PANGU_NO_COMPILE=1 retries
with eager mHC if capturing the compiled fns fails."""
import os, sys, time, torch
from exllamav3 import Config, Model, Cache, Tokenizer

MD = os.environ.get("PANGU_MD", "/root/models/openPangu-exl3")
REPLAYS = 30

config = Config.from_directory(MD)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 2048)
model.load()
tok = Tokenizer.from_config(config)

DEV = "cuda:0"
ids = tok.encode("The Industrial Revolution began in Great Britain in the late eighteenth "
                 "century and marked a turning point in history.", add_bos = True)
T0 = ids.shape[-1]
n_pages = (2048 + 255) // 256

# static device-resident params: no H2D, no host branch on the captured path
static_ids = torch.zeros((1, 1), dtype = torch.long, device = DEV)
static_bt = torch.arange(n_pages, dtype = torch.int32, device = DEV).unsqueeze(0)
static_sl = torch.zeros((1,), dtype = torch.int32, device = DEV)

def dev_params():
    return {
        "attn_mode": "flash_attn",
        "cache": cache,
        "block_table": static_bt,
        "cache_seqlens": static_sl,
        "conv_fresh": False,
    }

with torch.inference_mode():
    # prefill through the normal host-param path (registers conv slots)
    logits = model.forward(input_ids = ids, params = {
        "attn_mode": "flash_attn", "cache": cache,
        "block_table": torch.arange(n_pages, dtype = torch.int32).unsqueeze(0),
        "cache_seqlens": torch.tensor([0], dtype = torch.int32),
    })
    static_ids[0, 0] = int(logits[0, -1].argmax(-1).item())
    pos = T0

    # warmup: lazy builds (rope tables, sink kv, conv buffers, compiled fns)
    for _ in range(5):
        static_sl.fill_(pos)
        logits = model.forward(input_ids = static_ids, params = dev_params())
        static_ids[0, 0] = int(logits[0, -1].argmax(-1).item())
        pos += 1

    # timed eager (device params, no capture) for a like-for-like baseline
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        static_sl.fill_(pos)
        logits = model.forward(input_ids = static_ids, params = dev_params())
        pos += 1
    torch.cuda.synchronize()
    base = (time.time() - t0) / 10
    print(f"[EAGER-DEVPARAMS] {base * 1000:.1f} ms/token = {1 / base:.2f} tok/s")

    # capture
    static_sl.fill_(pos)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            static_out = model.forward(input_ids = static_ids, params = dev_params())
    except Exception as e:
        print("GRAPH CAPTURE BLOCKED:", repr(e)[:300])
        sys.exit(1)
    print("GRAPH CAPTURE OK")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(REPLAYS):
        static_sl += 1
        g.replay()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / REPLAYS
    print(f"[GRAPH REPLAY] {dt * 1000:.2f} ms/token = {1 / dt:.2f} tok/s over {REPLAYS} replays")
    print("sanity argmax of last replay:", int(static_out[0, -1].argmax(-1).item()))
