# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

import re
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
RDLogger.DisableLog('rdApp.*')


@register("multiroute_distill_cot")
class MultirouteDistillCoTRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        # Save our own compute_score method before calling parent __init__
        my_compute_score = self.compute_score
        
        # Call parent __init__ (this will set self.compute_score = None)
        super().__init__(config, tokenizer, None)
        
        # Restore our own compute_score method
        self.compute_score = my_compute_score
        
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.rollout_n = config.actor_rollout_ref.rollout.n
    
    def compute_score(self, data_source, solution_str, ground_truth, extra_info, **extra_reward_kwargs):
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, solution_str)
        if not matches or ("REACTANT" in matches[-1]):
            return -2.0, -1
        match_content = matches[-1]    

        try:
            match_content = Chem.MolToSmiles(Chem.MolFromSmiles(match_content), canonical=True)
        except:
            return -2.0, -1
        
        gt_list = [Chem.MolToSmiles(Chem.MolFromSmiles(gt), canonical=True) for gt in ground_truth]
        
        fpgen_gt_list = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(gt), 2, nBits=2048) for gt in gt_list]
        fpgen_match = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(match_content), 2, nBits=2048)
        tanimoto_scores = [DataStructs.TanimotoSimilarity(fpgen_match, fpgen_gt) for fpgen_gt in fpgen_gt_list]
        max_tanimoto_score = max(tanimoto_scores)
        loc = tanimoto_scores.index(max_tanimoto_score)
        if max_tanimoto_score >= 0.999:
            return 1.0, loc
        else:
            return -1.0, loc


    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == self.rollout_n, f"Should input a whole rollout results of a single query, data size {len(data)} != rollout_n {self.rollout_n}"
        
        ground_truth = data[0].non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data[0].non_tensor_batch["data_source"]
        extra_info = data[0].non_tensor_batch.get("extra_info", {})
        
        for i in range(self.rollout_n):
            data_item = data[i]
            
            assert data_item.non_tensor_batch["data_source"] == data_source, "Should be the same rollout Data source mismatch"
            assert data_item.non_tensor_batch["reward_model"]["ground_truth"] == ground_truth, "Should be the same rollout Ground truth mismatch"
            assert data_item.non_tensor_batch.get("extra_info", {}) == extra_info, "Should be the same rollout Extra info mismatch"
            
            assert len(data_item.non_tensor_batch["reward_model"]["ground_truth_answers"]) == len(ground_truth), "Ground truth answers size should be the same as ground truth size"
    
        assert len(ground_truth) <= self.rollout_n, "Ground truth size should be less than or equal to rollout_n"
        
        finded = [False for _ in range(len(ground_truth))]
        
        correct = [False for _ in range(self.rollout_n)]
        
        reward_scores = []
        reward_extra_info = {}
        
        # Collect all valid response ids for batch decoding
        all_valid_response_ids = []
        for i in range(self.rollout_n):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            all_valid_response_ids.append(valid_response_ids)
        
        # Batch decode all responses at once (faster with fast tokenizer)
        response_strs = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.batch_decode(all_valid_response_ids, skip_special_tokens=True)
        )
        
        drop = [False for _ in range(self.rollout_n)]
        
        for i in range(self.rollout_n):
            data_item = data[i]
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            response_str = response_strs[i]
        
            extra_reward_kwargs = (
                {
                    "reward_router_address": self.reward_router_address,
                    "reward_model_tokenizer": self.reward_model_tokenizer,
                }
                if self.reward_router_address is not None
                else {}
            )
            # compute_score is not async, call it directly
            result, loc = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key] = value
            else:
                score = result
                reward_extra_info["acc"] = score

            reward = score
            
            if loc != -1:
                if not finded[loc]:
                    correct[i] = True
                else:
                    correct[i] = False
                    drop[i] = True
                finded[loc] = True
                
            else:
                correct[i] = False
            reward_scores.append(reward)
        end_token_id = self.tokenizer.eos_token_id
        if sum(finded) != len(ground_truth):
            gt_answers = data[0].non_tensor_batch["reward_model"]["ground_truth_answers"]
            
            # Collect all missing gt_answers for batch encoding
            missing_gt_answers = [gt_answer for gt_answer, find in zip(gt_answers, finded) if not find]
            
            if missing_gt_answers:
                # Batch encode all missing answers at once (faster with fast tokenizer)
                encoded_results = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer(missing_gt_answers, add_special_tokens=False, return_tensors=None)
                )
                
                missing_idx = 0
                for gt_answer, find in zip(gt_answers, finded):
                    if find:
                        continue
                    
                    gt_answer_id_list = encoded_results['input_ids'][missing_idx]
                    gt_answer_id = torch.tensor(gt_answer_id_list + [end_token_id], dtype=torch.long)
                    missing_idx += 1
                    
                    # Find the first incorrect rollout to replace with this ground truth
                    for i in range(self.rollout_n):
                        if correct[i]:
                            continue
                        
                        prompt_ids = data[i].batch["prompts"]
                        prompt_length = prompt_ids.shape[-1]
                        response_length = gt_answer_id.shape[-1]
                        
                        reward_scores[i] = 1.0
                        data[i].batch["responses"] = gt_answer_id
                        data[i].batch["input_ids"] = torch.cat([prompt_ids, gt_answer_id])
                        data[i].batch["attention_mask"] = torch.ones(prompt_length + response_length, dtype=torch.long)
                        data[i].batch["position_ids"] = torch.arange(prompt_length + response_length, dtype=torch.long)
                        
                        response_mask = torch.zeros(prompt_length + response_length, dtype=torch.long)
                        response_mask[prompt_length:] = 1
                        data[i].batch["response_mask"] = response_mask
                        
                        correct[i] = True
                        if drop[i]:
                            drop[i] = False
                        break
        reward_extra_info = {}
        # reward_extra_info["reward_scores"] = reward_scores
        # reward_extra_info["correct"] = correct
        # reward_extra_info["finded"] = finded
        
        return {
            "reward_score": reward_scores,
            "reward_extra_info": reward_extra_info,
            "drop_mask": drop,  # Add drop mask for gradient masking
        }
