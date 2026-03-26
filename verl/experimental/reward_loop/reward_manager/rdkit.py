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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

@register("rdkit")
class RDKitRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    
    def _compute_rdkit_score(self, data_source, solution_str, ground_truth, extra_info, **extra_reward_kwargs):
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, solution_str)
        fpgen = AllChem.GetRDKitFPGenerator()
        if not match:
            return {"score": -1.0, "acc": 0.0, "tanimoto": -1.0}
        
        solution_smiles = match.group(1).strip()
        try:
            mol = Chem.MolFromSmiles(solution_smiles)
            if mol is None:
                return {"score": -1.0, "acc": 0.0, "tanimoto": -1.0}
            
            ans = Chem.MolToSmiles(mol, canonical=True)
            ans_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(ans))
            gt_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(ground_truth))
            
            similarity = DataStructs.TanimotoSimilarity(ans_fp, gt_fp)
            if similarity > 0.99:
                return {"score": 1.0, "acc": 1.0, "tanimoto": similarity}
            else:
                return {"score": 0.0, "acc": 0.0, "tanimoto": similarity}
                
        except Exception as e:
            # logger.warning(f"error: {e}")
            return {"score": -1.0, "acc": 0.0, "tanimoto": -1.0}
    
    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        # RDKit computation is CPU-bound, always run in executor
        result = await self.loop.run_in_executor(
            None,
            lambda: self._compute_rdkit_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            ),
        )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
