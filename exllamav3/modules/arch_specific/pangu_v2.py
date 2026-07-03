from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F
from ...model.config import Config
from ...modules import Module, Linear, RMSNorm, GatedMLP, BlockSparseMLP
from ..mla_attn import MLAAttention, _rms_std, _rms_kvc, _rotate_half
from ...util.tensor import get_for_device

# openPangu-2.0 (openpangu_v2) building blocks. Forward math mirrors Huawei's
# pure-torch reference (_pangu_torch_calib.py) exactly. The MLA core lives in
# modules/mla_attn.py; this file adds what is openPangu's own: MoME convs,
# learned sinks, the (Phase-2) DSA indexer, the mHC multi-stream residual and
# its decoder block. Phase 1: dense DSA, full-sequence forward, batch 1.


# mHC math as module-level pure fns so torch.compile can fuse each chain of
# tiny fp32 eager ops into a few kernels: 92 applications per token at batch-1
# decode are pure launch overhead otherwise. Op order matches the eager code
# exactly (fp32 throughout, same eps placement).

def _mhc_pre_merge(x, phi, norm_gamma, alpha, beta, eps, hc_eps, n, h):
    # x: [tok, N, H] -> collapsed [tok, H] (model-tail merge)
    flat = x.reshape(-1, n * h).float()
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + eps)
    mixes = F.linear(normed * norm_gamma, phi)
    h_pre = torch.sigmoid(mixes * alpha + beta.view(1, n)) + hc_eps
    return torch.sum(h_pre.view(-1, n, 1) * x.float(), dim = 1).to(x.dtype)


def _mhc_pre(x, phi, norm_gamma, alpha, beta, eps, hc_eps, n, h):
    # x: [tok, N, H] -> collapsed [tok, H] + post/res gates
    flat = x.reshape(-1, n * h).float()
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + eps)
    mixes = F.linear(normed * norm_gamma, phi)
    h_pre, h_post, h_res = mixes.split([n, n, n * n], dim = -1)
    a_pre, a_post, a_res = alpha.view(-1).split([1, 1, 1])
    b_pre, b_post, b_res = beta.view(-1).split([n, n, n * n])
    h_pre = torch.sigmoid(h_pre * a_pre + b_pre) + hc_eps
    h_post = 2 * torch.sigmoid(h_post * a_post + b_post)
    h_res = h_res.view(-1, n, n) * a_res + b_res.view(n, n)
    hidden = torch.sum(h_pre.view(-1, n, 1) * x.float(), dim = 1)
    return hidden.to(x.dtype), h_post, h_res


def _mhc_sinkhorn(h_res, recur_norm, hc_eps):
    h_res = h_res.float().softmax(-1) + hc_eps
    h_res = h_res / (h_res.sum(-2, keepdim = True) + hc_eps)
    for _ in range(max(recur_norm - 1, 0)):
        h_res = h_res / (h_res.sum(-1, keepdim = True) + hc_eps)
        h_res = h_res / (h_res.sum(-2, keepdim = True) + hc_eps)
    return h_res


def _mhc_post(x, h_post, residual, h_res, n, h):
    # x: [tok, H] sublayer output, residual: [tok, N, H] -> [tok, N, H]
    residual = residual.view(-1, n, h)
    hidden = (
        h_post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
        + torch.sum(h_res.float().unsqueeze(-1) * residual.float().unsqueeze(-2), dim = -3)
    )
    return hidden.to(x.dtype)


_mhc_pure = (_mhc_pre_merge, _mhc_pre, _mhc_sinkhorn, _mhc_post)
_mhc_fns = None

def _mhc(fn, *args):
    # Compile lazily on first call, dynamic = True so decode and prefill shapes
    # share graphs. EXLLAMA_PANGU_NO_COMPILE=1 or any compile/runtime failure
    # reverts to eager for the rest of the process.
    global _mhc_fns
    if _mhc_fns is None:
        if os.environ.get("EXLLAMA_PANGU_NO_COMPILE", None):
            _mhc_fns = {f: f for f in _mhc_pure}
        else:
            try:
                _mhc_fns = {f: torch.compile(f, dynamic = True) for f in _mhc_pure}
            except Exception:
                _mhc_fns = {f: f for f in _mhc_pure}
    g = _mhc_fns[fn]
    if g is fn:
        return fn(*args)
    try:
        return g(*args)
    except Exception:
        _mhc_fns = {f: f for f in _mhc_pure}
        return fn(*args)


class PanguMHC(Module):
    """Manifold-constrained hyper-connection: N parallel residual streams mixed by
    learned gates + a Sinkhorn-normalized NxN matrix. Replaces the residual add
    around each sublayer. All math in fp32. pre_only = the model-tail merge, which
    uses rms_norm_eps; the full path uses hc_eps = 1e-6."""

    def __init__(self, config, key, hidden_size, num_stream, rms_norm_eps, recur_norm, pre_only = False):
        super().__init__(config, key, None)
        self.module_name = "PanguMHC"
        self.hidden_size = hidden_size
        self.num_stream = num_stream
        self.recur_norm = recur_norm
        self.pre_only = pre_only
        self.hc_eps = 1e-6
        self.eps = rms_norm_eps if pre_only else self.hc_eps
        self.phi = None
        self.norm_gamma = None
        self.alpha = None
        self.beta = None

    @override
    def load(self, device, **kwargs):
        self.device = device
        # no_defer: these are transformed (.float()) at load, so a deferred
        # placeholder would be copied before it materializes
        get = lambda k: self.config.stc.get_tensor(f"{self.key}.{k}", device, float2half = True, no_defer = True).float()
        self.phi = get("phi.weight")
        self.norm_gamma = get("norm_gamma")
        if self.pre_only:
            self.alpha = get("branch_alpha_pre")
            self.beta = get("branch_beta_pre")
        else:
            self.alpha = get("branch_alpha")
            self.beta = get("branch_beta")

    @override
    def unload(self):
        self.device = None
        self.phi = self.norm_gamma = self.alpha = self.beta = None

    def optimizer_targets(self):
        return []

    def get_tensors(self):
        if self.device is None:
            return {}
        t = {f"{self.key}.phi.weight": self.phi.half(),
             f"{self.key}.norm_gamma": self.norm_gamma.half()}
        if self.pre_only:
            t[f"{self.key}.branch_alpha_pre"] = self.alpha.half()
            t[f"{self.key}.branch_beta_pre"] = self.beta.half()
        else:
            t[f"{self.key}.branch_alpha"] = self.alpha.half()
            t[f"{self.key}.branch_beta"] = self.beta.half()
        return t

    def mhc_pre(self, x):
        # x: [tok, N, H] -> collapsed [tok, H] (+ post/res gates on the full path)
        if self.pre_only:
            hidden = _mhc(_mhc_pre_merge, x, self.phi, self.norm_gamma, self.alpha, self.beta,
                          self.eps, self.hc_eps, self.num_stream, self.hidden_size)
            return hidden, None, None
        return _mhc(_mhc_pre, x, self.phi, self.norm_gamma, self.alpha, self.beta,
                    self.eps, self.hc_eps, self.num_stream, self.hidden_size)

    def mhc_sinkhorn(self, h_res):
        return _mhc(_mhc_sinkhorn, h_res, self.recur_norm, self.hc_eps)

    def mhc_post(self, x, h_post, residual, h_res):
        # x: [tok, H] sublayer output, residual: [tok, N, H] -> [tok, N, H]
        return _mhc(_mhc_post, x, h_post, residual, h_res, self.num_stream, self.hidden_size)

    def forward(self, x, params, out_dtype = None):
        # Merge module (model tail): collapse [b, s, N*H] -> [b, s, H]
        assert self.pre_only
        bsz, seqlen, _ = x.shape
        hidden, _, _ = self.mhc_pre(x.view(bsz * seqlen, self.num_stream, self.hidden_size))
        hidden = hidden.view(bsz, seqlen, self.hidden_size).half()
        if params.get("export_state_merge"):
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            s.append(hidden)
        return hidden


def _mome_conv(x, weight, kernel_width):
    # Depthwise causal conv1d over the token axis, fp32, +residual, first k-1
    # conv contributions zeroed (fresh sequence). x: [seq, dim]
    dtype = x.dtype
    seq = x.float()
    padded = F.pad(seq.transpose(0, 1).unsqueeze(0), (kernel_width - 1, 0))
    conv = F.conv1d(padded, weight, groups = weight.shape[0]).squeeze(0).transpose(0, 1)
    if kernel_width > 1:
        conv[: kernel_width - 1] = 0
    # contiguous: the transpose view otherwise reaches the EXL3 projections,
    # whose kernels assume row-major input (fp16 F.linear tolerates strides)
    return (conv + seq).to(dtype).contiguous()


_CONV_SLOTS = 1024

def _mome_conv_batched(x, prev, weight, kernel_width, pos):
    # Ring flavor of _mome_conv: x [bsz, q_len, dim], prev [bsz, k-1, dim] the
    # last k-1 pre-conv inputs per row (right-aligned), pos [bsz] tokens already
    # seen. The prev rows complete every window so no padding; conv contributions
    # at absolute positions < k-1 are dropped through where() (fresh-sequence
    # zeroing), which is also what masks windows reaching into prev rows not yet
    # written for short sequences
    k = kernel_width
    dtype = x.dtype
    seq = torch.cat((prev, x), dim = 1)
    f = seq.float()
    conv = F.conv1d(f.transpose(1, 2), weight, groups = weight.shape[0]).transpose(1, 2)
    apos = pos.unsqueeze(1) + torch.arange(x.shape[1], device = x.device).unsqueeze(0)
    conv = torch.where((apos >= k - 1).unsqueeze(-1), conv, 0.)
    return (conv + f[:, k - 1:]).to(dtype).contiguous(), seq[:, -(k - 1):]


class PanguAttention(MLAAttention):
    """MLA + openPangu extras: MoME depthwise convs on q_lora / kv_latent / attn
    out, learned always-visible sinks, and the DSA indexer weights (loaded for
    key coverage; sparsity is Phase 2). DSA layers run the absorbed path (dense
    == the reference topk path for seq_len <= index_topk)."""

    def __init__(self, config, key, layer_idx, is_dsa, sliding_window, qmap):
        super().__init__(
            config, key, layer_idx,
            hidden_size = config.hidden_size,
            num_heads = config.num_q_heads,
            q_lora_rank = config.q_lora_rank,
            kv_lora_rank = config.kv_lora_rank,
            qk_nope_head_dim = config.qk_nope_head_dim,
            qk_rope_head_dim = config.qk_rope_head_dim,
            v_head_dim = config.v_head_dim,
            rope_theta = config.rope_theta,
            rms_norm_eps = config.rms_norm_eps,
            qmap = qmap,
            sliding_window = None if is_dsa else sliding_window,
            absorbed = is_dsa,
            rope_interleave_pairs = False,
            kv_norm_dtype_mul = True,
        )
        self.module_name = "PanguAttention"
        self.is_dsa = is_dsa
        self.use_mome = config.use_mome and config.router_sliding_window > 0
        self.mome_kernel = config.router_sliding_window
        self.sink_count = config.param_sink_number
        if is_dsa:
            # Loaded for key coverage; unused until Phase-2 sparsity. Never quantized.
            self.idx_wq_b = Linear(config, f"{key}.indexer.wq_b", config.q_lora_rank,
                                   config.index_n_heads * config.index_head_dim, qmap = None)
            self.idx_wk = Linear(config, f"{key}.indexer.wk", config.hidden_size, config.index_head_dim, qmap = None)
            self.idx_weights = Linear(config, f"{key}.indexer.weights_proj", config.hidden_size,
                                      config.index_n_heads, qmap = None)
            for m in (self.idx_wq_b, self.idx_wk, self.idx_weights):
                self.register_submodule(m)
        self.idx_k_norm_w = None
        self.sink_ckv = None
        self.sink_kpe = None
        self.conv_qa = None
        self.conv_ckv = None
        self.conv_o = None
        # Paged decode state: up-projected sink K/V (built once per load), rope
        # tables built once to model max, per-site conv tail slot buffers, and
        # the page -> slot map, mirrored host-side (dict) and device-side
        # (table) so the decode path never reads it back
        self._sink_cache = None
        self._rope_max = min(config.max_position_embeddings, 131072)
        self._conv_state = {}
        self._slot_map = {}
        self._slot_page = [None] * _CONV_SLOTS
        self._slot_rr = 0
        self._page2slot = None

    @override
    def load_extra(self, device, get):
        if self.is_dsa:
            self.idx_k_norm_w = get("indexer.k_norm.weight")
        if self.sink_count > 0:
            self.sink_ckv = get("param_sink_compressed_kv")
            self.sink_kpe = get("param_sink_k_pe")
        if self.use_mome:
            self.conv_qa = get("qa_conv.weight").float()
            self.conv_ckv = get("compresskv_conv.weight").float()
            self.conv_o = get("o_conv.weight").float()

    @override
    def unload(self):
        super().unload()
        self.idx_k_norm_w = None
        self.sink_ckv = self.sink_kpe = None
        self.conv_qa = self.conv_ckv = self.conv_o = None
        self._sink_cache = None
        self._conv_state = {}
        self._slot_map = {}
        self._slot_page = [None] * _CONV_SLOTS
        self._slot_rr = 0
        self._page2slot = None

    @override
    def get_tensors_extra(self):
        t = {}
        if self.is_dsa:
            t[f"{self.key}.indexer.k_norm.weight"] = self.idx_k_norm_w
        if self.sink_count > 0:
            t[f"{self.key}.param_sink_compressed_kv"] = self.sink_ckv
            t[f"{self.key}.param_sink_k_pe"] = self.sink_kpe
        if self.use_mome:
            t[f"{self.key}.qa_conv.weight"] = self.conv_qa.half()
            t[f"{self.key}.compresskv_conv.weight"] = self.conv_ckv.half()
            t[f"{self.key}.o_conv.weight"] = self.conv_o.half()
        return t

    @override
    def hook_q_lora(self, q_lora):
        return _mome_conv(q_lora, self.conv_qa, self.mome_kernel) if self.use_mome else q_lora

    @override
    def hook_kv_latent(self, k_latent):
        return _mome_conv(k_latent, self.conv_ckv, self.mome_kernel) if self.use_mome else k_latent

    @override
    def hook_attn_out(self, attn):
        return _mome_conv(attn, self.conv_o, self.mome_kernel) if self.use_mome else attn

    @override
    def _sink_kv(self):
        # Sink K/V in cache layout ([heads, sinks, qk], V zero-padded), built
        # once per load. Sinks carry their learned rope part unrotated.
        if self._sink_cache is None:
            nope, v_dim = self.qk_nope_head_dim, self.v_head_dim
            lat_n = _rms_std(self.sink_ckv, self.kv_a_norm_w, self.rms_norm_eps)
            kv = self.kv_b_proj.forward(lat_n.unsqueeze(0), {})[0][..., : self.num_heads * (nope + v_dim)]
            kv = kv.view(self.sink_count, self.num_heads, nope + v_dim)
            k_nope, v = torch.split(kv, [nope, v_dim], dim = -1)
            k_pe = self.sink_kpe.to(k_nope.dtype).unsqueeze(1).expand(-1, self.num_heads, -1)
            k = torch.cat((k_nope, k_pe), dim = -1)
            v = F.pad(v, (0, self.qk_head_dim - v_dim))
            self._sink_cache = (k.transpose(0, 1).contiguous().float(),
                                v.transpose(0, 1).contiguous().float())
        return self._sink_cache

    def _conv_slot(self, page, device):
        # Slow path only. Round-robin slot reuse: a live sequence can only be
        # evicted past _CONV_SLOTS concurrent sequences. Newly assigned slots
        # are zeroed; a restart on an already mapped page keeps its slot, the
        # position mask covers the stale tail
        if page in self._slot_map:
            return
        s = self._slot_rr
        self._slot_rr = (s + 1) % _CONV_SLOTS
        old = self._slot_page[s]
        if old is not None:
            del self._slot_map[old]
        self._slot_page[s] = page
        self._slot_map[page] = s
        for st in self._conv_state.values():
            st[s].zero_()
        t = self._page2slot
        if t is None or page >= t.shape[0]:
            n = max(page + 1, 2 * t.shape[0] if t is not None else 4096)
            t2 = torch.zeros(n, device = device, dtype = torch.long)
            if t is not None:
                t2[: t.shape[0]] = t
            self._page2slot = t = t2
        t[page] = s

    def _conv_ring(self, site, x, weight, params):
        # Causal conv over [bsz, q_len, dim] with per-sequence tail state so
        # decode steps see the previous kernel_width - 1 pre-conv latents.
        # Sequences are keyed by their first cache page and enter through
        # cache_seqlens == 0, which restarts the row and maps its slot
        # host-side; past that the path is batched device ops, no host syncs
        k = self.mome_kernel
        state = self._conv_state.get(site)
        if state is None:
            state = torch.zeros(_CONV_SLOTS, k - 1, x.shape[-1], device = x.device, dtype = x.dtype)
            self._conv_state[site] = state
        seqlens_h = params["cache_seqlens"]  # host tensor from the generator
        if (seqlens_h == 0).any():
            bt_h = params["block_table"]
            for b in range(x.shape[0]):
                self._conv_slot(int(bt_h[b, 0]), x.device)
        bt = get_for_device(params, "block_table", x.device)
        seqlens = get_for_device(params, "cache_seqlens", x.device)
        slots = self._page2slot.index_select(0, bt[:, 0].long())
        prev = state.index_select(0, slots)
        out, tail = _mome_conv_batched(x, prev, weight, k, seqlens)
        state.index_copy_(0, slots, tail)
        return out

    @override
    def _fwd_paged(self, x, params):
        # Pangu flavor of the paged path: MoME convs run through ring state,
        # learned sinks merge into the flash output by logsumexp, DSA layers
        # attend dense over the cache (sliding_window is None for them)
        from flash_attn import flash_attn_with_kvcache
        from ...cache import Cache, CacheLayer
        bsz, q_len, hidden = x.shape
        nope, rope_d, v_dim = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim

        q_lora = self.q_a_proj.forward(x, params)[..., : self.q_lora_rank]
        if self.use_mome:
            q_lora = self._conv_ring("q", q_lora, self.conv_qa, params)
        q_lora = _rms_std(q_lora, self.q_a_norm_w, self.rms_norm_eps)
        q = self.q_b_proj.forward(q_lora, params)[..., : self.num_heads * self.qk_head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [nope, rope_d], dim = -1)

        kv = self.kv_a_proj.forward(x, params)[..., : self.kv_lora_rank + rope_d]
        k_latent, k_pe = torch.split(kv, [self.kv_lora_rank, rope_d], dim = -1)
        if self.use_mome:
            k_latent = self._conv_ring("kv", k_latent, self.conv_ckv, params)
        norm = _rms_kvc if self.kv_norm_dtype_mul else _rms_std
        k_lat_n = norm(k_latent, self.kv_a_norm_w, self.rms_norm_eps)
        kv_up = self.kv_b_proj.forward(k_lat_n, {})[..., : self.num_heads * (nope + v_dim)]
        kv_up = kv_up.view(bsz, q_len, self.num_heads, nope + v_dim)
        k_nope, v = torch.split(kv_up, [nope, v_dim], dim = -1)

        cache_seqlens = get_for_device(params, "cache_seqlens", x.device)
        block_table = get_for_device(params, "block_table", x.device)
        # rope tables build once to model max, then indexing by pos keeps the
        # decode step free of host syncs
        cos, sin = self._cos_sin(self._rope_max, x.device, q_pe.dtype)
        pos = cache_seqlens.long().unsqueeze(1) + torch.arange(q_len, device = x.device).unsqueeze(0)
        cos_q = cos[pos].unsqueeze(2)
        sin_q = sin[pos].unsqueeze(2)
        q_pe = ((q_pe.float() * cos_q.float()) + (_rotate_half(q_pe.float()) * sin_q.float())).to(q.dtype)
        cos_k = cos[pos].to(k_pe.dtype)
        sin_k = sin[pos].to(k_pe.dtype)
        k_pe = (k_pe * cos_k) + (_rotate_half(k_pe) * sin_k)

        q = torch.cat((q_nope, q_pe), dim = -1)
        k = torch.cat((k_nope, k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)), dim = -1)
        v = F.pad(v, (0, self.qk_head_dim - v_dim))

        cache = params.get("cache")
        instance = params.get("layer_instance")
        if isinstance(cache, CacheLayer):
            k_cache, v_cache = cache.get_kv(cache_seqlens, block_table, self.sliding_window)
        else:
            k_cache, v_cache = cache.get_layer(self.layer_idx, cache_seqlens, block_table, self.sliding_window, instance)
        window = (-1, -1) if self.sliding_window in (None, -1) else (self.sliding_window, 0)
        o, lse = flash_attn_with_kvcache(
            q = q.contiguous(),
            k_cache = k_cache,
            v_cache = v_cache,
            k = k.contiguous(),
            v = v.contiguous(),
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            causal = True,
            softmax_scale = self.scaling,
            window_size = window,
            return_softmax_lse = True,
        )
        if isinstance(cache, CacheLayer):
            cache.update_kv(cache_seqlens, block_table, k_cache, v_cache, q_len)
        else:
            cache.update_layer(self.layer_idx, cache_seqlens, block_table, k_cache, v_cache, q_len, instance)

        if self.sink_count > 0:
            sink_k, sink_v = self._sink_kv()
            scores = torch.einsum("bqhd,hsd->bqhs", q.float(), sink_k) * self.scaling
            lse_s = torch.logsumexp(scores, dim = -1)
            o_s = torch.einsum("bqhs,hsv->bqhv", torch.softmax(scores, dim = -1), sink_v)
            lse_c = lse.permute(0, 2, 1)
            m = torch.maximum(lse_c, lse_s)
            w_c = torch.exp(lse_c - m).unsqueeze(-1)
            w_s = torch.exp(lse_s - m).unsqueeze(-1)
            o = ((o.float() * w_c + o_s * w_s) / (w_c + w_s)).to(x.dtype)

        o = o[..., : v_dim].reshape(bsz, q_len, self.num_heads * v_dim)
        if self.use_mome:
            o = self._conv_ring("o", o, self.conv_o, params)
        return self.o_proj.forward(o, params)[..., : hidden]

    def extra_kv(self):
        if self.sink_count == 0:
            return None
        return _rms_std(self.sink_ckv, self.kv_a_norm_w, self.rms_norm_eps), self.sink_kpe


class PanguDecoderBlock(Module):
    """Decoder layer with mHC 4-stream residual, sandwich norms, and the optional
    block-post norm over the flattened stream carrier. Carries [b, s, N*H] between
    blocks; expands a [b, s, H] input (model entry) by stream repeat."""

    def __init__(self, config, key, layer_idx, attn, mlp, block_post_norm):
        super().__init__(config, key, None)
        self.module_name = "PanguDecoderBlock"
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_stream = config.mhc_num_stream
        self.attn = attn
        self.mlp = mlp
        self.attn_mhc = PanguMHC(config, f"{key}.attn_mhc_module", config.hidden_size,
                                 config.mhc_num_stream, config.rms_norm_eps, config.mhc_recur_norm)
        self.mlp_mhc = PanguMHC(config, f"{key}.mlp_mhc_module", config.hidden_size,
                                config.mhc_num_stream, config.rms_norm_eps, config.mhc_recur_norm)
        self.input_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(config, f"{key}.pre_mlp_layernorm", config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config, f"{key}.post_mlp_layernorm", config.rms_norm_eps)
        self.block_post_norm = block_post_norm
        for m in (self.attn_mhc, self.input_norm, self.attn, self.post_attn_norm,
                  self.mlp_mhc, self.pre_mlp_norm, self.mlp, self.post_mlp_norm,
                  self.block_post_norm):
            self.register_submodule(m)

    def optimizer_targets(self):
        t = []
        t += self.attn.optimizer_targets()
        t += self.mlp.optimizer_targets()
        return t

    def forward(self, x, params, out_dtype = None):
        n, h = self.num_stream, self.hidden_size
        bsz, seq_len, dim = x.shape
        paged = params.get("cache") is not None
        assert bsz == 1 or paged, "openpangu_v2 batch > 1 needs a paged cache"
        if x.dtype != torch.half:
            x = x.half()
        if dim == h:
            x = x.view(bsz * seq_len, 1, h).repeat(1, n, 1)
        else:
            x = x.view(bsz * seq_len, n, h)

        residual = x
        y, h_post, h_res = self.attn_mhc.mhc_pre(x)
        y = self.input_norm.forward(y, params, out_dtype = torch.half)
        if paged:
            y = self.attn.forward(y.view(bsz, seq_len, h), params).reshape(bsz * seq_len, h)
        else:
            y = self.attn.forward(y, params)
        y = self.post_attn_norm.forward(y, params, out_dtype = torch.half)
        h_res = self.attn_mhc.mhc_sinkhorn(h_res)
        x = self.attn_mhc.mhc_post(y, h_post, residual, h_res)

        residual = x
        y, h_post, h_res = self.mlp_mhc.mhc_pre(x)
        y = self.pre_mlp_norm.forward(y, params, out_dtype = torch.half)
        y = self.mlp.forward(y.view(1, -1, h), params)  # MLPs expect [bsz, q_len, dim]
        if y.dtype != torch.half:
            y = y.half()
        y = self.post_mlp_norm.forward(y.reshape(bsz * seq_len, h), params, out_dtype = torch.half)
        h_res = self.mlp_mhc.mhc_sinkhorn(h_res)
        x = self.mlp_mhc.mhc_post(y, h_post, residual, h_res)

        if self.block_post_norm is not None:
            x = self.block_post_norm.forward(x.reshape(bsz * seq_len, n * h), params, out_dtype = torch.half)
        return x.reshape(bsz, seq_len, n * h)
