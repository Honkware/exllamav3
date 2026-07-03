"""Pangu Generator validation: cached free-run speed, then MTP drafting with
per-depth dispatch. Token prefix identity vs the non-draft path + acceptance
stats + tok/s."""
import os, time, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MD = os.environ.get("PANGU_MD", "/root/models/openPangu-exl3")
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "The three most important inventions of the twentieth century were",
]
MAX_NEW = 96


def gen_all(use_mtp, ndt = None):
    config = Config.from_directory(MD)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 2048)
    tok = Tokenizer.from_config(config)

    # draft first: the trunk's autosplit takes whatever remains
    draft_model = None
    draft_cache = None
    if use_mtp:
        draft_model = Model.from_config(config, component = "mtp")
        draft_cache = Cache(draft_model, max_num_tokens = 2048)
        draft_model.load()
    model.load()
    if use_mtp:
        draft_model.attach_to(model)

    gen = Generator(model, cache, tok, draft_model = draft_model, draft_cache = draft_cache, num_draft_tokens = ndt)
    outs, stats = [], []
    for p in PROMPTS:
        job = Job(input_ids = tok.encode(p, add_bos = True), max_new_tokens = MAX_NEW,
                  sampler = GreedySampler())
        gen.enqueue(job)
        text = ""
        acc, rej, ntok = 0, 0, 0
        t0 = time.time()
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                text += r.get("text", "")
                acc = r.get("accepted_draft_tokens", acc)
                rej = r.get("rejected_draft_tokens", rej)
                ntok = r.get("new_tokens", ntok)
        dt = time.time() - t0
        outs.append(text)
        stats.append((acc, rej, ntok, dt))

    model.unload()
    if draft_model: draft_model.unload()
    del model, draft_model, cache, draft_cache, gen
    torch.cuda.empty_cache()
    return outs, stats


base, bstats = gen_all(False)
for i, (a, s) in enumerate(zip(base, bstats)):
    print(f"[base] prompt {i}: {s[2] or MAX_NEW} tok in {s[3]:.1f}s = {(s[2] or MAX_NEW) / s[3]:.2f} tok/s")
    print("   ", repr(a[:120]))

for ndt in (1, 3):
    mtp, stats = gen_all(True, ndt)
    ok = True
    for i, (a, b) in enumerate(zip(base, mtp)):
        match = a.startswith(b) or b.startswith(a)
        ok = ok and match
        acc, rej, ntok, dt = stats[i]
        rate = acc / max(acc + rej, 1)
        tps = (ntok or MAX_NEW) / dt
        print(f"[ndt={ndt}] [{'OK  ' if match else 'DIFF'}] prompt {i}: acc {acc} rej {rej} rate {rate:.1%} | {tps:.2f} tok/s")
        if not match:
            d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print("  base:", repr(a[max(0, d - 40):d + 40]))
            print("  mtp: ", repr(b[max(0, d - 40):d + 40]))
    print(f"[ndt={ndt}] TOKEN PREFIX IDENTITY:", "PASS" if ok else "FAIL")
