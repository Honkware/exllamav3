from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ...model.config import Config
from ...modules import Module, Linear, RMSNorm, GatedMLP, BlockSparseMLP
from ..mla_attn import MLAAttention, _rms_std

# openPangu-2.0 (openpangu_v2) building blocks. Forward math mirrors Huawei's
# pure-torch reference (_pangu_torch_calib.py) exactly. The MLA core lives in
# modules/mla_attn.py; this file adds what is openPangu's own: MoME convs,
# learned sinks, the (Phase-2) DSA indexer, the mHC multi-stream residual and
# its decoder block. Phase 1: dense DSA, full-sequence forward, batch 1.


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
        n, h = self.num_stream, self.hidden_size
        dtype = x.dtype
        flat = x.reshape(-1, n * h).float()
        normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + self.eps)
        mixes = F.linear(normed * self.norm_gamma, self.phi)
        if self.pre_only:
            h_pre = torch.sigmoid(mixes * self.alpha + self.beta.view(1, n)) + self.hc_eps
            hidden = torch.sum(h_pre.view(-1, n, 1) * x.float(), dim = 1)
            return hidden.to(dtype), None, None
        h_pre, h_post, h_res = mixes.split([n, n, n * n], dim = -1)
        a_pre, a_post, a_res = self.alpha.view(-1).split([1, 1, 1])
        b_pre, b_post, b_res = self.beta.view(-1).split([n, n, n * n])
        h_pre = torch.sigmoid(h_pre * a_pre + b_pre) + self.hc_eps
        h_post = 2 * torch.sigmoid(h_post * a_post + b_post)
        h_res = h_res.view(-1, n, n) * a_res + b_res.view(n, n)
        hidden = torch.sum(h_pre.view(-1, n, 1) * x.float(), dim = 1)
        return hidden.to(dtype), h_post, h_res

    def mhc_sinkhorn(self, h_res):
        h_res = h_res.float().softmax(-1) + self.hc_eps
        h_res = h_res / (h_res.sum(-2, keepdim = True) + self.hc_eps)
        for _ in range(max(self.recur_norm - 1, 0)):
            h_res = h_res / (h_res.sum(-1, keepdim = True) + self.hc_eps)
            h_res = h_res / (h_res.sum(-2, keepdim = True) + self.hc_eps)
        return h_res

    def mhc_post(self, x, h_post, residual, h_res):
        # x: [tok, H] sublayer output, residual: [tok, N, H] -> [tok, N, H]
        n, h = self.num_stream, self.hidden_size
        dtype = x.dtype
        residual = residual.view(-1, n, h)
        hidden = (
            h_post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
            + torch.sum(h_res.float().unsqueeze(-1) * residual.float().unsqueeze(-2), dim = -3)
        )
        return hidden.to(dtype)

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
    return (conv + seq).to(dtype)


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
        assert bsz == 1, "openpangu_v2 Phase 1 runs batch size 1"
        if x.dtype != torch.half:
            x = x.half()
        if dim == h:
            x = x.view(bsz * seq_len, 1, h).repeat(1, n, 1)
        else:
            x = x.view(bsz * seq_len, n, h)

        residual = x
        y, h_post, h_res = self.attn_mhc.mhc_pre(x)
        y = self.input_norm.forward(y, params, out_dtype = torch.half)
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
