from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, GatedMLP, BlockSparseMLP, Linear
from ..modules.arch_specific.pangu_v2 import PanguAttention
from ..modules.arch_specific.mimo_mtp import MiMoMTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .openpangu_v2 import OpenPanguV2Config

# openPangu ships three independent DeepSeek-style MTP depth heads at layers
# 46..48, each self-contained: own enorm/hnorm/eh_proj fusion, one
# pangu-flavored block (MoME convs + sinks, SWA 2048, MoE; no mHC), own
# shared_head.norm. The per-depth embed_tokens and shared_head.head are tied
# to the trunk's (verified on real weights) and borrowed via attach_to()
# rather than converted. All depths quantize; forward runs the depth selected
# by params["mtp_step"] (default 0), matching the drafting loop's chained
# target_hidden dataflow. Generator-side per-step dispatch lands with the
# pangu paged decode path.


class PanguMTPInputLayer(MiMoMTPInputLayer):
    # DeepSeek concat order: (embedding, hidden), the reverse of MiMo

    def forward(self, x, params, out_dtype = None):
        from ..util.tensor import get_for_device, to2
        target_hidden = params.get("target_hidden")
        assert target_hidden is not None, "MTP requires target_hidden"
        y = get_for_device(params, "target_hidden", self.device)
        y = self.norm_hidden.forward(y, params)
        if not self.attached_model().loaded_tp:
            x = self.attached_model().modules[0].forward(x, params, out_dtype = torch.half)
        else:
            raise NotImplementedError()
        x = self.norm_embedding.forward(x.to(self.device), params)
        x = torch.cat((x, y.to(x.device)), dim = -1)
        x = self.proj.forward(x, params)
        return to2(x, out_dtype, self.out_dtype)


class OpenPanguV2MTPModel(Model):

    def __init__(
        self,
        config: OpenPanguV2Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.num_depths = config.num_nextn_predict_layers
        self.modules = []
        self.depth_slices = []
        self.head_norms = []
        self.input_layers = []

        for depth in range(self.num_depths):
            idx = config.num_hidden_layers + depth
            key = f"model.layers.{idx}"

            # module key must differ from the block's: the convert stage saves
            # per-module tensor files named by key, and identical keys clobber
            input_layer = PanguMTPInputLayer(
                config = config,
                key = f"{key}.mtp_in",
                key_norm_hidden = f"{key}.hnorm",
                key_norm_embedding = f"{key}.enorm",
                key_proj = f"{key}.eh_proj",
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                native_draft_len = 1,
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
            )
            attn = PanguAttention(
                config, f"{key}.self_attn",
                layer_idx = depth,
                is_dsa = False,
                sliding_window = 2048,
                qmap = "block.attn",
            )
            block = TransformerBlock(
                config = config,
                key = key,
                layer_idx = depth,
                attn_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps),
                attn = attn,
                attn_post_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps),
                mlp_norm = RMSNorm(config, f"{key}.pre_mlp_layernorm", config.rms_norm_eps),
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
                ),
                mlp_post_norm = RMSNorm(config, f"{key}.post_mlp_layernorm", config.rms_norm_eps),
                qbits_key = "mtp_bits",
            )
            # Own norm weights per depth (not tied to model.norm); the head is
            # tied to lm_head and borrowed, so it is not a module here
            head_norm = RMSNorm(
                config = config,
                key = f"{key}.shared_head.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            )
            first = len(self.modules)
            self.modules += [input_layer, block, head_norm]
            self.depth_slices.append((first, first + 2))
            self.input_layers.append(input_layer)
            self.head_norms.append(head_norm)

        self.first_block_idx = 0
        self.last_kv_module_idx = self.depth_slices[-1][1] - 1

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": self.num_depths,
            "autosplit_load_fwd": False,
        })

        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)


    @override
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        # Run the depth selected by the drafting step. Chain state stays
        # pre-head-norm; the norm and borrowed head apply in sample_from_state
        if params is None:
            params = {}
        step = min(params.get("mtp_step", 0), self.num_depths - 1)
        a, b = self.depth_slices[step]
        x = self.prepare_inputs(input_ids, params)
        for module in self.modules[a: b]:
            params["layer_instance"] = 0
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    @override
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        # Refresh depth-0's cache for accepted positions. Deeper heads keep
        # their drafting-time states; unwritten positions read back as zeros
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        a, b = self.depth_slices[0]
        for module in self.modules[a: b]:
            params["layer_instance"] = 0
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)


    def attach_to(self, target):
        # Borrow the target's embedding and lm_head (per-depth embed_tokens and
        # shared_head.head are tied to the trunk's). The trunk exports the
        # post-merge pre-norm hidden for target_hidden
        for il in self.input_layers:
            il.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)
        self.draft_verifier_params.update({
            "export_state_merge": True,
        })


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        step = min(params.get("mtp_step", 0), self.num_depths - 1)
        norm = self.head_norms[step]
        state = norm.forward(state.to(norm.device), params, out_dtype = torch.half)
        assert not self.attached_model().loaded_tp, "openpangu_v2 MTP does not support TP"
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state, params)
        logits = lm.forward(logits, params)
        return torch.argmax(logits, dim = -1)
