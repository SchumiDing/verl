# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = None  # Will be set by FSDPWorker
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False, collect_attention_scores: bool = False  
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        
        # # Debug: Log response_length and check if it's valid
        # if torch.distributed.get_rank() == 0 and collect_attention_scores:
        #     print(f"\n[DEBUG] Response info:")
        #     print(f"  response_length from micro_batch['responses'].size(-1): {response_length}")
        #     print(f"  micro_batch['responses'].shape: {micro_batch['responses'].shape}")
        #     if "response_mask" in micro_batch:
        #         print(f"  micro_batch['response_mask'].shape: {micro_batch['response_mask'].shape}")
        #         print(f"  response_mask sum: {micro_batch['response_mask'].sum().item()}")
        
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            response = micro_batch["responses"]
            batch_size, seqlen = input_ids.shape
            # Save original batch_size and seqlen for pad_input later
            # These should NOT be modified during the forward pass
            original_batch_size = batch_size
            original_seqlen = seqlen
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            cu_seqlens_for_collector = None
            original_attention_mask_for_collector = attention_mask.clone()  # Save original attention_mask
            
            # Create token mask BEFORE remove_padding to maintain correct dimensions
            batch_size_orig, seqlen_orig = input_ids.shape
            token_mask = torch.zeros(batch_size_orig, seqlen_orig, dtype=torch.long, device=input_ids.device)
            
            if collect_attention_scores:
                # # Debug: check if input_ids contains valid tokens
                # if torch.distributed.get_rank() == 0:
                #     first_sample = input_ids[0]
                #     first_mask = attention_mask[0]
                #     valid_len_check = int(first_mask.sum().item())
                #     # Check first 10 and last 10 valid tokens
                #     print(f"\n[DEBUG] Input IDs check for first sample:")
                #     print(f"  valid_length: {valid_len_check}")
                #     # print(f"  reponse: {response[0].tolist()}")
                #     print(f"  first sample: {first_sample.tolist()}")
                #     print(f"  First 10 tokens: {first_sample[:10].tolist()}")
                #     print(f"  Last 10 tokens: {first_sample[valid_len_check-10:valid_len_check].tolist()}")
                #     print(f"  PAD token ID: {self.tokenizer.pad_token_id if self.tokenizer else 'N/A'}")
                #     print(f"  EOS token ID: {self.tokenizer.eos_token_id if self.tokenizer else 'N/A'}")
                
                # Fill token_mask based on response_mask or response_length
                if "response_mask" in micro_batch:
                    # Multi-turn case: response_mask marks ALL assistant tokens
                    response_mask_full = micro_batch["response_mask"]
                    response_mask_len = response_mask_full.shape[1]
                    
                    # # Debug: check response_mask content
                    # if torch.distributed.get_rank() == 0:
                    #     print(f"\n[DEBUG] Response mask check:")
                    #     print(f"  response_mask shape: {response_mask_full.shape}")
                    #     print(f"  First sample response_mask sum: {response_mask_full[0].sum().item()}")
                    #     # Find where response tokens are
                    #     response_indices = torch.nonzero(response_mask_full[0], as_tuple=False).squeeze(-1)
                    #     if len(response_indices) > 0:
                    #         print(f"  Response token positions (first/last 5): {response_indices[:5].tolist()} ... {response_indices[-5:].tolist()}")
                    #     else:
                    #         print(f"  WARNING: No response tokens marked in response_mask!")
                    
                    if response_mask_len == seqlen_orig:
                        # Same length: directly copy (excluding last token)
                        token_mask[:, :-1] = response_mask_full[:, :-1]
                    elif response_mask_len < seqlen_orig:
                        # response_mask is shorter: map to actual response positions
                        # For each sample, find where the response actually starts based on valid_length
                        # The response_mask is aligned with the actual response content, not the full padded sequence
                        for b in range(batch_size_orig):
                            valid_length = int(attention_mask[b].sum().item())
                            # Add bounds check
                            if valid_length > seqlen_orig or valid_length < 0:
                                if torch.distributed.get_rank() == 0:
                                    print(f"[Warning] Invalid valid_length={valid_length} for seqlen={seqlen_orig}, sample {b}")
                                valid_length = min(max(valid_length, 0), seqlen_orig)
                            
                            # Get the actual response length for this sample from response_mask
                            actual_response_length = int(response_mask_full[b].sum().item())
                            
                            if actual_response_length > 0 and valid_length > 0:
                                # The response is at the end of the valid sequence
                                prompt_length = valid_length - actual_response_length
                                # Add bounds check for prompt_length
                                if prompt_length < 0:
                                    if torch.distributed.get_rank() == 0:
                                        print(f"[Warning] Negative prompt_length={prompt_length} for sample {b}, actual_response_length={actual_response_length}, valid_length={valid_length}")
                                    continue
                                
                                if valid_length > 0:
                                    # Map response_mask to the actual positions in the full sequence
                                    # Copy only the valid response tokens (excluding the last token for shift)
                                    response_end = min(valid_length - 1, seqlen_orig)
                                    response_start = prompt_length
                                    mask_tokens_to_copy = min(actual_response_length - 1, response_end - response_start)
                                    
                                    if mask_tokens_to_copy > 0 and response_start >= 0 and response_start < seqlen_orig:
                                        # Find which positions in response_mask are actually 1
                                        response_mask_indices = torch.nonzero(response_mask_full[b], as_tuple=False).squeeze(-1)
                                        if len(response_mask_indices) > 0:
                                            # Take at most mask_tokens_to_copy indices (excluding last)
                                            indices_to_use = response_mask_indices[:mask_tokens_to_copy]
                                            # Map these to positions in token_mask
                                            for idx, mask_idx in enumerate(indices_to_use):
                                                target_pos = response_start + idx
                                                if target_pos < seqlen_orig and mask_idx < response_mask_len:
                                                    token_mask[b, target_pos] = response_mask_full[b, mask_idx]
                    else:
                        if torch.distributed.get_rank() == 0:
                            print(f"[Warning] response_mask_len {response_mask_len} > seqlen {seqlen_orig}, truncating")
                        # Truncate response_mask to fit
                        token_mask[:, :-1] = response_mask_full[:, :seqlen_orig-1]
                else:
                    # Single-turn case: use response_length to identify assistant tokens
                    response_length = micro_batch["responses"].size(-1)
                    for b in range(batch_size_orig):
                        valid_length = int(attention_mask[b].sum().item())
                        # Add bounds check
                        if valid_length > seqlen_orig or valid_length < 0:
                            if torch.distributed.get_rank() == 0:
                                print(f"[Warning] Invalid valid_length={valid_length} for seqlen={seqlen_orig}, sample {b}")
                            valid_length = min(max(valid_length, 0), seqlen_orig)
                        
                        prompt_length = valid_length - response_length
                        # Add bounds check
                        if prompt_length < 0:
                            if torch.distributed.get_rank() == 0:
                                print(f"[Warning] Negative prompt_length={prompt_length} for sample {b}, skipping")
                            continue
                        
                        if valid_length > 1:  # Need at least 2 tokens
                            end_pos = min(valid_length - 1, seqlen_orig)
                            if end_pos > prompt_length:
                                token_mask[b, prompt_length:end_pos] = 1
            
            # Log token_mask statistics
            if torch.distributed.get_rank() == 0 and collect_attention_scores:
                # print(f"\n[STEP 0] After token_mask creation:")
                # print(f"  token_mask.shape: {token_mask.shape}")
                # print(f"  token_mask.sum(): {token_mask.sum().item()}")
                # print(f"  Samples with response tokens: {(token_mask.sum(dim=1) > 0).sum().item()}/{batch_size_orig}")
                # print(f"  Original batch_size: {original_batch_size}, original_seqlen: {original_seqlen}")
                # Show per-sample statistics
                for b in range(min(3, batch_size_orig)):  # Show first 3 samples
                    valid_len = int(attention_mask[b].sum().item())
                    response_tokens = int(token_mask[b].sum().item())
                    # Calculate prompt_length correctly
                    actual_prompt_length = valid_len - response_tokens
                    # print(f"  Sample {b}: valid_length={valid_len}, response_tokens={response_tokens}, prompt_length={actual_prompt_length}")
            
            if self.use_remove_padding:
                # Log input shapes before unpad_input
                # if torch.distributed.get_rank() == 0 and collect_attention_scores:
                #     print(f"\n[STEP 1] Before unpad_input:")
                #     print(f"  input_ids.shape: {input_ids.shape}")
                #     print(f"  attention_mask.shape: {attention_mask.shape}")
                #     print(f"  attention_mask.sum(): {attention_mask.sum().item()}")
                #     print(f"  batch_size: {batch_size}, seqlen: {seqlen}")
                
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                
                # # Log indices information after unpad_input
                # if torch.distributed.get_rank() == 0 and collect_attention_scores:
                #     print(f"\n[STEP 2] After unpad_input:")
                #     print(f"  input_ids_rmpad.shape: {input_ids_rmpad.shape}")
                #     print(f"  indices.shape: {indices.shape}")
                #     print(f"  indices.min(): {indices.min().item()}, indices.max(): {indices.max().item()}")
                #     print(f"  cu_seqlens.shape: {cu_seqlens.shape}")
                #     print(f"  cu_seqlens: {cu_seqlens.tolist()}")
                #     print(f"  Expected max index (batch_size * seqlen - 1): {batch_size * seqlen - 1}")
                #     if indices.max().item() >= batch_size * seqlen:
                #         print(f"  [WARNING] indices.max() >= batch_size * seqlen!")
                
                # Store cu_seqlens for attention score collection
                cu_seqlens_for_collector = cu_seqlens

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                
                # Add safety check for rolled input_ids
                if collect_attention_scores and torch.distributed.get_rank() == 0:
                    # Check if any labels are out of bounds
                    vocab_size = getattr(self.actor_module.config, 'vocab_size', 151936)  # Qwen2 default
                    max_label = input_ids_rmpad_rolled.max().item()
                    min_label = input_ids_rmpad_rolled.min().item()
                    if max_label >= vocab_size or min_label < 0:
                        print(f"[WARNING] Labels out of bounds!")
                        print(f"  vocab_size: {vocab_size}")
                        print(f"  min_label: {min_label}, max_label: {max_label}")


                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                if collect_attention_scores:
                    from verl.models.transformers.attention_score_patch import get_attention_collector
                    collector = get_attention_collector()
                    collector.enable()
                    collector.reset()
                    
                    # Pass the pre-created token_mask, cu_seqlens, and original attention_mask
                    cu_seqlens_for_mask = cu_seqlens_for_collector if self.use_remove_padding else None
                    attention_mask_for_collector = original_attention_mask_for_collector if self.use_remove_padding else attention_mask
                    collector.set_token_mask(token_mask, cu_seqlens=cu_seqlens_for_mask, attention_mask=attention_mask_for_collector)
                    
                    # # Log first sample's dialogue on rank 0
                    # if torch.distributed.get_rank() == 0 and self.tokenizer is not None:
                    #     try:
                    #         # Use original input_ids and masks before padding removal
                    #         first_sample_ids = micro_batch["input_ids"][0]
                    #         first_sample_mask = original_attention_mask_for_collector[0]
                    #         first_token_mask = token_mask[0]
                            
                    #         # Get valid length
                    #         valid_length = int(first_sample_mask.sum().item())
                            
                    #         # Find the range of valid tokens (non-padding) to handle left/right padding
                    #         valid_indices = torch.nonzero(first_sample_mask, as_tuple=False).squeeze(-1)
                    #         if len(valid_indices) == 0:
                    #             print("[WARNING] No valid tokens found in first sample!")
                    #             raise ValueError("No valid tokens found")
                            
                    #         start_idx = int(valid_indices[0].item())
                    #         end_idx = int(valid_indices[-1].item()) + 1
                            
                    #         # Decode full sequence (only valid tokens)
                    #         valid_tokens = first_sample_ids[start_idx:end_idx]
                    #         full_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
                            
                    #         # For multi-turn: separate into segments based on token_mask changes
                    #         prompt_indices = []
                    #         response_indices = []
                            
                    #         in_response = False
                    #         for idx in range(start_idx, end_idx):
                    #             is_response = first_token_mask[idx].item() == 1
                    #             if is_response:
                    #                 response_indices.append(idx)
                    #                 in_response = True
                    #             else:
                    #                 prompt_indices.append(idx)
                    #                 if in_response:
                    #                     # Transition from response to prompt (new turn)
                    #                     in_response = False
                            
                    #         num_response_tokens = len(response_indices)
                    #         num_prompt_tokens = len(prompt_indices)
                            
                    #         print(f"\n{'='*80}")
                    #         print(f"[Rank 0] First sample dialogue sequence:")
                    #         print(f"  Total tokens: {valid_length}")
                    #         print(f"  Valid token range: [{start_idx}, {end_idx})")
                    #         print(f"  Prompt tokens: {num_prompt_tokens}")
                    #         print(f"  Response tokens (assistant): {num_response_tokens}")
                    #         print(f"{'-'*80}")
                    #         print(f"FULL CONVERSATION:")
                    #         print(full_text)
                    #         print(f"{'='*80}\n")
                    #     except Exception as e:
                    #         import traceback
                    #         print(f"[Rank 0] Failed to decode dialogue: {e}")
                    #         traceback.print_exc()
                            
                # if torch.distributed.get_rank() == 0:
                #     print(f"Before actor module:")
                #     print(f"input_ids_rmpad: {input_ids_rmpad.shape}")
                #     print(f"First sample: {self.tokenizer.decode(input_ids_rmpad[0].tolist(), skip_special_tokens=False)}")
                    
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                attention_scores_dict = None
                if collect_attention_scores:
                    collector = get_attention_collector()
                    if len(collector.top_scores_per_layer) > 0:
                        top_scores = torch.stack(collector.top_scores_per_layer)  # (num_layers * batch_size,)
                        bottom_scores = torch.stack(collector.bottom_scores_per_layer)  # (num_layers * batch_size,)
                        
                        # Reshape to (num_layers, batch_size) for easier analysis
                        try:
                            num_layers = self.actor_module.config.num_hidden_layers
                        except:
                            num_layers = self.actor_module.config.text_config.num_hidden_layers
                        
                        batch_size = top_scores.shape[0] // num_layers
                        top_scores = top_scores.reshape(num_layers, batch_size)
                        bottom_scores = bottom_scores.reshape(num_layers, batch_size)
                        
                        # Store scores for logging
                        attention_scores_dict = {
                            "top_scores_mean": top_scores.mean().item(),
                            "bottom_scores_mean": bottom_scores.mean().item(),
                            "top_scores_std": top_scores.std().item(),
                            "bottom_scores_std": bottom_scores.std().item(),
                            "top_scores_per_layer_mean": top_scores.mean(dim=1).detach(),  # (num_layers,)
                            "bottom_scores_per_layer_mean": bottom_scores.mean(dim=1).detach(),  # (num_layers,)
                        }
                        
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    
                    # Add immediate check after logprobs computation
                    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                        if torch.distributed.get_rank() == 0:
                            print(f"[ERROR] Invalid log_probs immediately after logprobs_from_logits")
                            print(f"  NaN: {torch.isnan(log_probs).any()}, Inf: {torch.isinf(log_probs).any()}")
                            print(f"  log_probs shape: {log_probs.shape}")
                            print(f"  log_probs stats: min={log_probs.min()}, max={log_probs.max()}")


                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                # Log tensor shapes before pad_input
                # if torch.distributed.get_rank() == 0 and collect_attention_scores:
                #     print(f"\n[STEP 3] Before pad_input:")
                #     print(f"  log_probs.shape: {log_probs.shape}")
                #     if calculate_entropy:
                #         print(f"  entropy_rmpad.shape: {entropy_rmpad.shape}")
                #     if calculate_sum_pi_squared:
                #         print(f"  sum_pi_squared_rmpad.shape: {sum_pi_squared_rmpad.shape}")
                #     print(f"  indices.shape: {indices.shape}")
                #     print(f"  indices.min(): {indices.min().item()}, indices.max(): {indices.max().item()}")
                #     print(f"  Using original_batch_size: {original_batch_size}, original_seqlen: {original_seqlen}")
                #     print(f"  original_batch_size * original_seqlen: {original_batch_size * original_seqlen}")
                #     if indices.max().item() >= original_batch_size * original_seqlen:
                #         print(f"  [ERROR] indices.max() ({indices.max().item()}) >= original_batch_size * original_seqlen ({original_batch_size * original_seqlen})!")
                #         print(f"  This will cause index out of bounds in pad_input!")
                
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=original_batch_size,
                        seqlen=original_seqlen,
                    )
                    # Fix padding positions for entropy (should be 0, which is actually correct for entropy)
                    # Entropy of a deterministic distribution is 0, so padding=0 is acceptable
                    
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=original_batch_size,
                        seqlen=original_seqlen,
                    )
                    # Fix padding positions for sum_pi_squared (should be 0, which is correct)

                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=original_batch_size,
                    seqlen=original_seqlen,
                )
                
                # # Log tensor shapes after pad_input
                # if torch.distributed.get_rank() == 0 and collect_attention_scores:
                #     print(f"\n[STEP 4] After pad_input:")
                #     print(f"  full_log_probs.shape: {full_log_probs.shape}")
                #     if calculate_entropy:
                #         print(f"  full_entropy.shape: {full_entropy.shape}")
                #     if calculate_sum_pi_squared:
                #         print(f"  full_sum_pi_squared.shape: {full_sum_pi_squared.shape}")
                #     print(f"  Expected shape: ({original_batch_size}, {original_seqlen}, 1)")
                #     print(f"  attention_mask.shape: {attention_mask.shape}")
                
                # CRITICAL FIX: pad_input initializes padding positions to 0, but log_probs should be negative
                # Replace 0 values (padding) with a safe negative value
                # This prevents issues when these values are used in exp() calculations
                # Use try-except to catch any CUDA errors from previous operations
                try:
                    full_log_probs_2d = full_log_probs.squeeze(-1)
                    # Create padding mask safely - move to CPU if needed to avoid CUDA errors
                    padding_mask = (full_log_probs_2d == 0) & (attention_mask == 0)  # True for padding positions
                    
                    # Check if there are any padding positions to fix
                    has_padding = padding_mask.any().item()  # Force sync here
                    
                    if has_padding:
                        full_log_probs_2d = full_log_probs_2d.masked_fill(padding_mask, 0.0)
                        full_log_probs = full_log_probs_2d.unsqueeze(-1)
                        # if torch.distributed.get_rank() == 0 and collect_attention_scores:
                        #     num_fixed = padding_mask.sum().item()
                        #     print(f"  [Debug] Fixed {num_fixed} padding positions in log_probs (set to 0.0)")
                except RuntimeError as e:
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        if torch.distributed.get_rank() == 0:
                            print(f"[ERROR] CUDA error detected during log_probs padding fix: {e}")
                            print(f"  This indicates an earlier CUDA assertion was triggered")
                            print(f"  full_log_probs shape: {full_log_probs.shape}")
                            print(f"  attention_mask shape: {attention_mask.shape}")
                        # Re-raise to propagate the error
                        raise



                # only return response part:
                # Note: In multi-turn scenarios with response_mask, we should use the mask
                # to extract responses instead of assuming they're at the end
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                # # Add debug info for log_probs extraction
                # if collect_attention_scores and torch.distributed.get_rank() == 0:
                #     print(f"[Debug] After padding back:")
                #     print(f"  full_log_probs shape: {full_log_probs.shape}")
                #     print(f"  response_length: {response_length}")
                #     print(f"  extracted log_probs shape: {log_probs.shape}")
                #     print(f"  log_probs min/max/mean: {log_probs.min():.4f}/{log_probs.max():.4f}/{log_probs.mean():.4f}")
                #     # Check for zeros in log_probs (which shouldn't be there)
                #     num_zeros = (log_probs == 0).sum().item()
                #     if num_zeros > 0:
                #         print(f"  [WARNING] Found {num_zeros} zero values in log_probs!")


            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                if collect_attention_scores:
                    from verl.models.transformers.attention_score_patch import get_attention_collector
                    collector = get_attention_collector()
                    collector.enable()
                    collector.reset()
                    
                    # Pass the pre-created token_mask (no cu_seqlens in non-rmpad mode)
                    collector.set_token_mask(token_mask, cu_seqlens=None, attention_mask=attention_mask)
                    
                    # Log first sample's dialogue on rank 0
                    # if torch.distributed.get_rank() == 0 and self.tokenizer is not None:
                    #     try:
                    #         first_sample_ids = input_ids[0]
                    #         first_sample_mask = attention_mask[0]
                    #         first_token_mask = token_mask[0]
                            
                    #         # Get valid length
                    #         valid_length = int(first_sample_mask.sum().item())
                            
                    #         # Find the range of valid tokens (non-padding) to handle left/right padding
                    #         valid_indices = torch.nonzero(first_sample_mask, as_tuple=False).squeeze(-1)
                    #         if len(valid_indices) == 0:
                    #             print("[WARNING] No valid tokens found in first sample!")
                    #             raise ValueError("No valid tokens found")
                            
                    #         start_idx = int(valid_indices[0].item())
                    #         end_idx = int(valid_indices[-1].item()) + 1
                            
                    #         # Decode full sequence (only valid tokens)
                    #         valid_tokens = first_sample_ids[start_idx:end_idx]
                    #         full_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
                            
                    #         # For multi-turn: separate into segments based on token_mask changes
                    #         prompt_indices = []
                    #         response_indices = []
                            
                    #         in_response = False
                    #         for idx in range(start_idx, end_idx):
                    #             is_response = first_token_mask[idx].item() == 1
                    #             if is_response:
                    #                 response_indices.append(idx)
                    #                 in_response = True
                    #             else:
                    #                 prompt_indices.append(idx)
                    #                 if in_response:
                    #                     # Transition from response to prompt (new turn)
                    #                     in_response = False
                            
                    #         num_response_tokens = len(response_indices)
                    #         num_prompt_tokens = len(prompt_indices)
                            
                    #         print(f"\n{'='*80}")
                    #         print(f"[Rank 0] First sample dialogue sequence (non-rmpad mode):")
                    #         print(f"  Total tokens: {valid_length}")
                    #         print(f"  Valid token range: [{start_idx}, {end_idx})")
                    #         print(f"  Prompt tokens: {num_prompt_tokens}")
                    #         print(f"  Response tokens (assistant): {num_response_tokens}")
                    #         print(f"{'-'*80}")
                    #         print(f"FULL CONVERSATION:")
                    #         print(full_text)
                    #         print(f"{'='*80}\n")
                    #     except Exception as e:
                    #         import traceback
                    #         print(f"[Rank 0] Failed to decode dialogue: {e}")
                    #         traceback.print_exc()

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                attention_scores_dict = None
                if collect_attention_scores:
                    collector = get_attention_collector()
                    if len(collector.top_scores_per_layer) > 0:
                        top_scores = torch.stack(collector.top_scores_per_layer)
                        bottom_scores = torch.stack(collector.bottom_scores_per_layer)
                        
                        # Reshape to (num_layers, batch_size)
                        try:
                            num_layers = self.actor_module.config.num_hidden_layers
                        except:
                            num_layers = self.actor_module.config.text_config.num_hidden_layers
                        
                        batch_size = top_scores.shape[0] // num_layers
                        top_scores = top_scores.reshape(num_layers, batch_size)
                        bottom_scores = bottom_scores.reshape(num_layers, batch_size)
                        
                        attention_scores_dict = {
                            "top_scores_mean": top_scores.mean().item(),
                            "bottom_scores_mean": bottom_scores.mean().item(),
                            "top_scores_std": top_scores.std().item(),
                            "bottom_scores_std": bottom_scores.std().item(),
                            "top_scores_per_layer_mean": top_scores.mean(dim=1).detach(),
                            "bottom_scores_per_layer_mean": bottom_scores.mean(dim=1).detach(),
                        }

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            # Add safety check before returning
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                if torch.distributed.get_rank() == 0:
                    print(f"[ERROR] Invalid log_probs in _forward_micro_batch (rmpad path)")
                    print(f"  NaN={torch.isnan(log_probs).any()}, Inf={torch.isinf(log_probs).any()}")
                    print(f"  log_probs stats: min={log_probs.min()}, max={log_probs.max()}, mean={log_probs.mean()}")
            
            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if attention_scores_dict is not None:
                outputs["attention_scores"] = attention_scores_dict
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    collect_attention_scores = self.config.get("collect_attention_scores", False)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                        collect_attention_scores=collect_attention_scores
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    
                    # Add safety check for log_prob
                    if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                        if torch.distributed.get_rank() == 0:
                            print(f"[ERROR] Invalid log_prob detected: NaN={torch.isnan(log_prob).any()}, Inf={torch.isinf(log_prob).any()}")
                            print(f"  log_prob stats: min={log_prob.min()}, max={log_prob.max()}, mean={log_prob.mean()}")
                        # Replace invalid values with a safe value
                        log_prob = torch.nan_to_num(log_prob, nan=-100.0, posinf=-100.0, neginf=-100.0)
                    
                    # # CRITICAL: Check dimension matching between log_prob and response_mask
                    # if log_prob.shape != response_mask.shape:
                    #     if torch.distributed.get_rank() == 0:
                    #         print(f"[ERROR] Shape mismatch!")
                    #         print(f"  log_prob.shape: {log_prob.shape}")
                    #         print(f"  response_mask.shape: {response_mask.shape}")
                    #         print(f"  This will cause incorrect masking in loss computation!")
                    
                    # Check if there are zero values in log_prob where response_mask is 1
                    # if (log_prob == 0).any():
                    #     zero_in_response = ((log_prob == 0) & (response_mask == 1)).sum().item()
                    #     if zero_in_response > 0 and torch.distributed.get_rank() == 0:
                    #         print(f"[ERROR] Found {zero_in_response} zero log_prob values in response positions!")
                    #         print(f"  This indicates padding positions are not properly excluded!")
                    #         # Show which positions have this problem
                    #         for b in range(min(2, log_prob.shape[0])):  # Check first 2 samples
                    #             bad_positions = torch.where((log_prob[b] == 0) & (response_mask[b] == 1))[0]
                    #             if len(bad_positions) > 0:
                    #                 print(f"    Sample {b}: positions {bad_positions.tolist()[:10]}")  # Show first 10

                    
                    # Record attention scores if available
                    if "attention_scores" in outputs:
                        attn_scores = outputs["attention_scores"]
                        micro_batch_metrics["actor/attn_top_mean"] = attn_scores["top_scores_mean"]
                        micro_batch_metrics["actor/attn_bottom_mean"] = attn_scores["bottom_scores_mean"]
                        micro_batch_metrics["actor/attn_top_std"] = attn_scores["top_scores_std"]
                        micro_batch_metrics["actor/attn_bottom_std"] = attn_scores["bottom_scores_std"]

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
