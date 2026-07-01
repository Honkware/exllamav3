"""Gate 9 round-trip: load the converted Moonlight EXL3 quant, greedy-generate
via the Phase-1 no-cache loop (full re-forward per token), decode, check
coherence + stop behavior."""
import json, os, sys
import torch

QD = "/root/models/Moonlight-exl3"

@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Tokenizer
    config = Config.from_directory(QD)
    model = Model.from_config(config)
    model.load()
    tok = Tokenizer.from_config(config)

    eos = set()
    for f in ("generation_config.json", "config.json"):
        try:
            d = json.load(open(os.path.join(QD, f)))
            e = d.get("eos_token_id")
            if isinstance(e, int): eos.add(e)
            if isinstance(e, list): eos.update(e)
        except Exception:
            pass
    print("eos set:", eos)

    for prompt in ("The capital of France is", "def fibonacci(n):"):
        ids = tok.encode(prompt)
        if ids.dim() == 1: ids = ids.unsqueeze(0)
        for _ in range(40):
            x = model.modules[0].forward(ids, {}).to("cuda:0")
            for m in model.modules[1:]:
                if x.dtype != torch.half: x = x.half()
                x = m.forward(x, {})
            nxt = x[0, -1].argmax().item()
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim = -1)
            if nxt in eos:
                print("(hit eos)", end = " ")
                break
        print("PROMPT:", repr(prompt))
        print("OUTPUT:", repr(tok.decode(ids[0])))
        print()

if __name__ == "__main__":
    main()
