# Batch-1 decode benchmark for openPangu-2.0 EXL3 (exl3-pangu2 fork).
# Run on the GPU pod. Sections run independently:
#   (a) 30-step end-to-end decode timing
#   (b) torch.profiler over 5 steps: top kernels/ops by self CUDA time, launch counts
#   (c) CUDA-graph capture smoke test of one decode forward
#
#   PANGU_MD=/path/to/model python3 bench_decode.py [a|b|c ...]

import os, sys, time, traceback
import torch
from exllamav3 import Config, Model, Cache, Tokenizer

MD = os.environ.get("PANGU_MD", "/root/models/openPangu-exl3")
MAX_TOKENS = 2048
PROMPT = ("The quick brown fox jumps over the lazy dog. " * 12).strip()
PROMPT_LEN = 50
WARMUP = 5
STEPS_A = 30
STEPS_B = 5
REPLAYS_C = 30


def hr(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


class St:
    # shared decode state: model, cache, current position, next input ids
    pass


def make_params(st, seqlens = None, bt = None):
    return {
        "attn_mode": "flash_attn",
        "cache": st.cache,
        "block_table": st.bt if bt is None else bt,
        "cache_seqlens": torch.tensor([st.pos], dtype = torch.int32) if seqlens is None else seqlens,
    }


def decode_step(st):
    # one greedy decode step, ids stay on device
    logits = st.model.forward(input_ids = st.ids, params = make_params(st))
    st.ids = logits[:, -1, :].argmax(dim = -1, keepdim = True)
    st.pos += 1


def setup():
    st = St()
    print(f"Loading {MD} ...")
    config = Config.from_directory(MD)
    st.model = Model.from_config(config)
    st.cache = Cache(st.model, max_num_tokens = MAX_TOKENS)
    st.model.load(progressbar = True)
    st.tok = Tokenizer.from_config(config)
    st.dev = st.model.modules[-1].device or "cuda:0"
    st.bt = torch.arange(MAX_TOKENS // 256, dtype = torch.int32).unsqueeze(0)

    ids = st.tok.encode(PROMPT)
    ids = ids[:, :PROMPT_LEN]
    print(f"Prefill: {ids.shape[-1]} tokens")
    st.pos = 0
    logits = st.model.forward(input_ids = ids, params = make_params(st))
    st.pos = ids.shape[-1]
    st.ids = logits[:, -1:, :].argmax(dim = -1)
    torch.cuda.synchronize()
    return st


def sec_a(st):
    hr("(a) end-to-end decode timing")
    for _ in range(WARMUP):
        decode_step(st)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS_A):
        decode_step(st)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = dt / STEPS_A * 1000
    print(f"{STEPS_A} decode steps: {dt:.3f} s total, {ms:.2f} ms/token, {1000 / ms:.2f} tok/s")


def sec_b(st):
    hr(f"(b) torch.profiler over {STEPS_B} decode steps")
    from torch.profiler import profile, ProfilerActivity
    decode_step(st)  # warm
    torch.cuda.synchronize()
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(STEPS_B):
            decode_step(st)
        torch.cuda.synchronize()

    ka = prof.key_averages()
    print("\n--- top-15 ops by self CUDA time ---")
    try:
        print(ka.table(sort_by = "self_cuda_time_total", row_limit = 15))
    except Exception:
        print(ka.table(sort_by = "self_device_time_total", row_limit = 15))

    # aggregate device-side kernel events by name
    kern = {}
    n_launch_rt = 0
    for e in prof.events():
        name = getattr(e, "key", None) or getattr(e, "name", "?")
        if name == "cudaLaunchKernel":
            n_launch_rt += 1
        dt_ = str(getattr(e, "device_type", ""))
        if dt_.endswith("CUDA"):
            t = getattr(e, "self_device_time_total", None)
            if t is None:
                t = getattr(e, "self_cuda_time_total", 0.0)
            c, s = kern.get(name, (0, 0.0))
            kern[name] = (c + 1, s + float(t or 0.0))

    print("\n--- top-15 CUDA kernels by self CUDA time ---")
    rows = sorted(kern.items(), key = lambda kv: -kv[1][1])[:15]
    if rows:
        print(f"{'us total':>12} {'count':>8} {'count/step':>11}  kernel")
        for name, (c, s) in rows:
            print(f"{s:12.1f} {c:8d} {c / STEPS_B:11.1f}  {name[:100]}")
    else:
        print("(no device-side kernel events found in this torch version)")

    n_kernels = sum(c for c, _ in kern.values())
    n_launch = n_launch_rt or n_kernels
    print(f"\nkernel launches: {n_launch} total over {STEPS_B} steps = {n_launch / STEPS_B:.0f} per step"
          + ("" if n_launch_rt else " (from device events; no cudaLaunchKernel rows)"))

    cpu_self = sum(getattr(r, "self_cpu_time_total", 0.0) for r in ka)
    cuda_self = sum(float(getattr(r, "self_device_time_total", None)
                          or getattr(r, "self_cuda_time_total", 0.0) or 0.0) for r in ka)
    print(f"sum self CPU time (launch/dispatch overhead): {cpu_self / 1000:.1f} ms "
          f"({cpu_self / 1000 / STEPS_B:.1f} ms/step)")
    print(f"sum self CUDA time: {cuda_self / 1000:.1f} ms ({cuda_self / 1000 / STEPS_B:.1f} ms/step)")


def sec_c(st):
    hr("(c) CUDA-graph capture smoke test")
    # static buffers; cache_seqlens lives on device and is incremented in place
    dev = st.ids.device if st.ids.is_cuda else st.dev
    static_ids = st.ids.to(dev).clone()
    static_bt = st.bt.to(dev)
    static_seqlens = torch.tensor([st.pos], dtype = torch.int32, device = dev)

    def fwd():
        # fresh params dict each call (per-forward dev_cache), static tensors inside
        return st.model.forward(
            input_ids = static_ids,
            params = {
                "attn_mode": "flash_attn",
                "cache": st.cache,
                "block_table": static_bt,
                "cache_seqlens": static_seqlens,
            },
        )

    try:
        # warmup on a side stream, as required before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fwd()
                static_seqlens += 1
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_logits = fwd()
        torch.cuda.synchronize()

        for _ in range(3):  # replay warmup
            static_seqlens += 1
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPLAYS_C):
            static_seqlens += 1
            g.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / REPLAYS_C * 1000
        print(f"GRAPH CAPTURE OK, replay ms/token = {ms:.2f} over {REPLAYS_C} replays "
              f"({1000 / ms:.1f} tok/s, forward only, logits {tuple(static_logits.shape)})")
    except Exception as e:
        print("GRAPH CAPTURE BLOCKED")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    sections = [a for a in sys.argv[1:] if a in ("a", "b", "c")] or ["a", "b", "c"]
    st = setup()
    for sec, fn in (("a", sec_a), ("b", sec_b), ("c", sec_c)):
        if sec not in sections:
            continue
        try:
            fn(st)
        except Exception:
            print(f"--- section ({sec}) failed ---")
            traceback.print_exc()


if __name__ == "__main__":
    main()
