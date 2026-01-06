import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.utils.log_utils import init_logger
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.qwen3next.mem_manager import Qwen3NextHybridMemManager
from lightllm.models.qwen3next.infer_struct import Qwen3NextFlashAttentionStateInfo
from typing import Tuple
from typing_extensions import override
from einops import rearrange
from lightllm.models.qwen3next.triton_kernel.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
from lightllm.models.qwen3next.triton_kernel.fla.ops import chunk_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.fla.ops import fused_recurrent_gated_delta_rule
from lightllm.common.basemodel.layer_infer.transformer_layer_infer import TransformerLayerInfer
from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward
from lightllm.utils.envs_utils import get_env_start_args
from functools import partial

logger = init_logger(__name__)


class Qwen3NextFullAttentionTransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)
        super().__init__(layer_num, network_config, mode)

    @override
    def _att_norm(
        self,
        input,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.att_norm_weight_.weight, self.eps_, out=out)
        return out

    @override
    def _ffn_norm(
        self,
        input,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.ffn_norm_weight_.weight, self.eps_, out=out)
        return out

    @override
    def _bind_norm(self):
        self._att_norm = partial(Qwen3NextFullAttentionTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(Qwen3NextFullAttentionTransformerLayerInfer._ffn_norm, self)
        return

    @override
    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        # save for get_o gatedly
        infer_state.gate_value = torch.sigmoid(layer_weight.o_gate_proj.mm(input))
        cache_kv = layer_weight.kv_proj.mm(
            input.view(-1, self.embed_dim_),
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        gemma_rmsnorm_forward(
            q.view(-1, self.head_dim_),
            layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = gemma_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        return q, cache_kv

    @override
    def _get_o(
        self,
        input,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        # Handle different input shapes from different attention kernels
        input = input.view(-1, infer_state.gate_value.shape[-1])
        gated_input = input * infer_state.gate_value
        infer_state.gate_value = None
        o_tensor = layer_weight.o_proj.mm(gated_input)
        return o_tensor

    @override
    def _bind_ffn(self):
        super()._bind_ffn()
        self._original_ffn = self._ffn
        self._original_tpsp_ffn = self._tpsp_ffn
        self._ffn = partial(Qwen3NextFullAttentionTransformerLayerInfer._ffn_with_shared_expert, self)

    def _ffn_with_shared_expert(
        self,
        input,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        shared_expert_out = F.sigmoid(layer_weight.shared_expert_gate.mm(input)) * ffn2_out
        moe_out = self._original_ffn(input, infer_state, layer_weight)
        return shared_expert_out + moe_out


class Qwen3NextGatedDeltaNetTransformerLayerInfer(Qwen3NextFullAttentionTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.network_config_ = network_config
        self.embed_dim_ = self.network_config_["hidden_size"]
        self.num_v_heads = self.network_config_["linear_num_value_heads"]
        self.num_k_heads = self.network_config_["linear_num_key_heads"]
        self.head_k_dim = self.network_config_["linear_key_head_dim"]
        self.head_v_dim = self.network_config_["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = self.network_config_["linear_conv_kernel_dim"]
        self.activation = self.network_config_["hidden_act"]
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // self.tp_world_size_
        self.tp_ba_dim = (self.num_v_heads * 2) // self.tp_world_size_
        self.tp_num_k_heads = self.num_k_heads // self.tp_world_size_
        self.tp_num_v_heads = self.num_v_heads // self.tp_world_size_
        self.tp_key_dim = self.key_dim // self.tp_world_size_
        self.tp_value_dim = self.value_dim // self.tp_world_size_
        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads
        self.mtp_step = get_env_start_args().mtp_step
        self.mtp_size = self.mtp_step + 1
        return

    def _fix_query_key_value_ba_ordering(self, mixed_qkvzba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        mixed_qkvz, mixed_ba = torch.split(mixed_qkvzba, [self.tp_qkvz_dim, self.tp_ba_dim], dim=-1)

        mixed_qkvz = mixed_qkvz.view(
            -1,
            self.tp_num_k_heads,
            self.head_k_dim + self.head_k_dim + (self.head_v_dim + self.head_v_dim) * self.num_v_heads_per_k_head,
        )
        mixed_ba = mixed_ba.view(-1, self.tp_num_k_heads, 2 * self.num_v_heads_per_k_head)

        qkvz_split_list = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads_per_k_head * self.head_v_dim),
            (self.num_v_heads_per_k_head * self.head_v_dim),
        ]
        (query, key, value, z) = torch.split(mixed_qkvz, qkvz_split_list, dim=2)
        (b, a) = torch.split(mixed_ba, [self.num_v_heads_per_k_head, self.num_v_heads_per_k_head], dim=2)

        query = query.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        key = key.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        value = value.reshape(-1, self.tp_num_v_heads * self.head_v_dim)
        z = z.reshape(-1, self.tp_num_v_heads, self.head_v_dim)
        b = b.reshape(-1, self.tp_num_v_heads)
        a = a.reshape(-1, self.tp_num_v_heads)

        return query, key, value, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
            dim=-1,
        )
        if decode:
            batch_size = mixed_qkv.shape[0]
            query = query.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
        else:
            query, key = map(lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim), (query, key))
            value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query, key, value

    @override
    def context_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=True)
        return gdn_out

    @override
    def token_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=False)
        return gdn_out

    def _gdn_prefill_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Prefill kernel for GDN forward pass."""
        # Conv1D processing
        mixed_qkv = mixed_qkv.transpose(0, 1)
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.mm_param.bias,
            query_start_loc=infer_state.b1_cu_q_seq_len,
            cache_indices=infer_state.b_buffer_idx,
            has_initial_state=infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=self.activation,
        )
        mixed_qkv = out_tensor.transpose(0, 1)

        # Recurrent processing
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv)
        initial_state = ssm_states[infer_state.b_buffer_idx]
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states[infer_state.b_buffer_idx] = last_recurrent_state.to(ssm_states.dtype, copy=False)
        return core_attn_out

    def _gdn_decode_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Decode kernel for GDN forward pass (single-token, non-MTP mode)."""
        # Conv1D processing
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.mm_param.bias,
            activation=self.activation,
            conv_state_indices=infer_state.b_buffer_idx,
        )

        # Recurrent processing
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv, decode=True)
        # g and beta have shape (1, batch, num_heads), need to squeeze and unsqueeze to get (batch, 1, num_heads)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g.squeeze(0).unsqueeze(1),
            beta=beta.squeeze(0).unsqueeze(1),
            initial_state=ssm_states,
            inplace_final_state=True,
            ssm_state_indices=infer_state.b_buffer_idx,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out

    def _gdn_decode_mtp_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Decode kernel for GDN forward pass (MTP mode with multiple steps)."""
        # Conv1D processing for all MTP steps
        for step_idx in range(self.mtp_size):
            mixed_qkv_i = mixed_qkv[step_idx :: self.mtp_size].contiguous()
            cur_buffer_idx = infer_state.mtp_buffer_idx_list[step_idx]
            mixed_qkv[step_idx :: self.mtp_size].copy_(
                causal_conv1d_update(
                    mixed_qkv_i,
                    conv_states,
                    layer_weight.linear_conv1d.mm_param.weight,
                    bias=layer_weight.linear_conv1d.mm_param.bias,
                    activation=self.activation,
                    conv_state_indices=cur_buffer_idx,
                )
            )
            if step_idx != self.mtp_step:
                next_buffer_idx = infer_state.mtp_buffer_idx_list[step_idx + 1]
                conv_states[next_buffer_idx] = conv_states[cur_buffer_idx].clone()

        # Recurrent processing for all MTP steps
        g_squeezed = g.squeeze(0)  # (batch, num_heads)
        beta_squeezed = beta.squeeze(0)  # (batch, num_heads)

        core_attn_out = torch.empty(
            (mixed_qkv.shape[0], 1, self.tp_num_v_heads, self.head_v_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )

        for step_idx in range(self.mtp_size):
            mixed_qkv_i = mixed_qkv[step_idx :: self.mtp_size].contiguous()
            query_i, key_i, value_i = self._rearrange_mixed_qkv(mixed_qkv_i, decode=True)

            # Extract g and beta for this step from the pre-computed values
            g_i = g_squeezed[step_idx :: self.mtp_size].unsqueeze(1)
            beta_i = beta_squeezed[step_idx :: self.mtp_size].unsqueeze(1)

            cur_buffer_idx = infer_state.mtp_buffer_idx_list[step_idx]

            core_attn_out_i, _ = fused_recurrent_gated_delta_rule(
                q=query_i,
                k=key_i,
                v=value_i,
                g=g_i,
                beta=beta_i,
                initial_state=ssm_states,
                inplace_final_state=True,
                ssm_state_indices=cur_buffer_idx,
                use_qk_l2norm_in_kernel=True,
            )

            core_attn_out[step_idx :: self.mtp_size].copy_(core_attn_out_i)

            if step_idx != self.mtp_step:
                next_buffer_idx = infer_state.mtp_buffer_idx_list[step_idx + 1]
                ssm_states[next_buffer_idx] = ssm_states[cur_buffer_idx].clone()

        return core_attn_out

    def gdn_forward(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextFlashAttentionStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
        is_prefill: bool,
    ):
        assert isinstance(infer_state.mem_manager, Qwen3NextHybridMemManager)

        # Common preprocessing
        input = input.view(-1, self.embed_dim_)
        conv_states, ssm_states = infer_state.mem_manager.get_mamba_cache(self.layer_num_)

        mixed_qkvzba = layer_weight.linear_in_proj.mm(input)
        q, k, v, z, b, a = self._fix_query_key_value_ba_ordering(mixed_qkvzba)
        mixed_qkv = torch.cat([q, k, v], dim=-1)

        # Compute g and beta for all modes
        g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)

        # Dispatch to appropriate kernel
        if is_prefill:
            core_attn_out = self._gdn_prefill_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )
        elif self.mtp_step == 0:
            core_attn_out = self._gdn_decode_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )
        else:
            core_attn_out = self._gdn_decode_mtp_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )

        # Common postprocessing
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = self.alloc_tensor(core_attn_out.shape, core_attn_out.dtype, device=core_attn_out.device)
        gated_rmsnorm_forward(
            core_attn_out,
            layer_weight.linear_norm.weight,
            layer_weight.linear_norm.bias,
            self.eps_,
            z,
            out=norm_out,
        )
        core_attn_out = norm_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output = layer_weight.linear_out_proj.mm(core_attn_out)
        if self.tp_world_size_ > 1:
            all_reduce(output, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return output
