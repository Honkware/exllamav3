from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import RMSNorm, Embedding, GatedMLP, BlockSparseMLP, Linear
from ..modules.arch_specific.pangu_v2 import PanguAttention, PanguDecoderBlock, PanguMHC
from .openpangu_v2_mtp import OpenPanguV2MTPModel
from ..modules.attn import prepare_for_attn

# openPangu-2.0 (DeepSeek-lineage MoE with MLA, mHC 4-stream residual, MoME convs,
# learned attention sinks, interleaved DSA/SWA layers, sandwich norms). Phase 1:
# dense forward + EXL3 conversion; DSA layers run dense, MTP layers (46..48)
# are not loaded.

class OpenPanguV2Config(Config):
    arch_string = "OpenPanguV2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": OpenPanguV2Model, "mtp": OpenPanguV2MTPModel},
            **kwargs
        )

        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)

        # MLA
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", no_default)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        self.rope_theta = self.read_cfg(float, "rope_theta", 10000.0)

        # Layer typing
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.dsa_layers = set(self.read_cfg(list, "dsa_layers", []))
        self.swa_layers = self.read_cfg(list, "swa_layers", [])
        self.sliding_window_list = self.read_cfg(list, "sliding_window_list", [])
        self.block_post_layernorm_idx = set(self.read_cfg(list, "block_post_layernorm_idx", []))

        # DSA indexer (weights loaded, gating deferred to Phase 2)
        self.index_topk = self.read_cfg(int, "index_topk", 0)
        self.index_n_heads = self.read_cfg(int, "index_n_heads", 0)
        self.index_head_dim = self.read_cfg(int, "index_head_dim", 0)

        # Sinks / MoME / mHC
        self.param_sink_number = self.read_cfg(int, "param_sink_number", 0)
        self.use_mome = self.read_cfg(bool, "use_mome", False)
        self.router_sliding_window = self.read_cfg(int, "router_sliding_window", 0)
        self.use_mhc = self.read_cfg(bool, "use_mhc", False)
        self.mhc_num_stream = self.read_cfg(int, "mhc_num_stream", 1)
        self.mhc_recur_norm = self.read_cfg(int, "mhc_recur_norm", 1)
        assert self.use_mhc and self.mhc_num_stream > 1, \
            "openpangu_v2 support assumes the mHC multi-stream path"
        self.assert_cfg(bool, "sandwich_norm", True, True)

        # MLP / MoE
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "n_shared_experts", 1)
        self.num_experts = self.read_cfg(int, "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", 0)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # MTP depth heads (layers 46..48), loadable as draft components
        self.num_nextn_predict_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        if self.num_nextn_predict_layers == 0:
            del self.model_classes["mtp"]


class OpenPanguV2Model(Model):
    config_class = OpenPanguV2Config

    def __init__(
        self,
        config: OpenPanguV2Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.caps.update({"supports_tp": False})

        self.modules += [
            Embedding(
                config = config,
                key = "model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)
        swa_pos = {idx: i for i, idx in enumerate(config.swa_layers)}

        for idx in range(config.num_hidden_layers):
            key = f"model.layers.{idx}"
            is_dsa = idx in config.dsa_layers
            window = config.sliding_window_list[swa_pos[idx]] if idx in swa_pos else None

            attn = PanguAttention(
                config, f"{key}.self_attn",
                layer_idx = idx,
                is_dsa = is_dsa,
                sliding_window = None if is_dsa else window,
                qmap = "block.attn",
            )
            if idx < config.first_k_dense_replace:
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                )
            else:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    key_e_score_bias = "e_score_correction_bias",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    router_type = "ds3",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = 1,
                    topk_group = 1,
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"{key}.mlp.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            block_post_norm = None
            if idx in config.block_post_layernorm_idx:
                block_post_norm = RMSNorm(
                    config = config,
                    key = f"{key}.block_post_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                )
            self.modules += [
                PanguDecoderBlock(config, key, idx, attn, mlp, block_post_norm)
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = "model.embed_tokens"

        self.modules += [
            PanguMHC(
                config, "model.merge_mhc_module", config.hidden_size,
                config.mhc_num_stream, config.rms_norm_eps, config.mhc_recur_norm,
                pre_only = True,
            ),
            RMSNorm(
                config = config,
                key = "model.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<|pangu_text_start|>"
        if system_prompt:
            p += f"<|message_start|>system\n{system_prompt}<|message_end|>"
        p += f"<|message_start|>user\n{prompt}<|message_end|>"
        p += f"<|message_start|>assistant\n"
        return p
