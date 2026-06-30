import logging
import os
import time

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _compute_response_mask_from_attention(data: DataProto) -> torch.Tensor:
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


class BagelRollout(BaseRollout):
    """In-process BAGEL rollout that shares weights with the actor module."""

    def __init__(self, actor_module, tokenizer, rollout_config):
        # This in-process rollout is used by BAGEL actor integration and does
        # not own a separate inference engine.
        self.device_mesh = None
        self.actor_module = actor_module
        self.tokenizer = tokenizer
        self.config: RolloutConfig = omega_conf_to_dataclass(rollout_config)

    async def resume(self, tags: list[str]):
        return None

    async def update_weights(self, weights, **kwargs):
        return None

    async def release(self):
        return None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        started = time.time()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)

        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        multi_modal_data_batch = prompts.non_tensor_batch.get("multi_modal_data", None)

        self.actor_module.eval()
        with torch.no_grad():
            responses_list = self.actor_module(
                mode="generate",
                raw_prompts=raw_prompts,
                multi_modal_data_batch=multi_modal_data_batch,
                max_length=response_length,
                do_sample=do_sample,
                temperature=temperature,
            )

        response = torch.full(
            (batch_size, response_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=idx.dtype,
            device=idx.device,
        )
        for i, response_ids in enumerate(responses_list):
            if response_ids.numel() == 0:
                continue
            valid_len = min(response_ids.numel(), response_length)
            response[i, :valid_len] = response_ids[:valid_len].to(device=idx.device, dtype=idx.dtype)

        seq = torch.cat([idx, response], dim=-1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(
            batch_size, -1
        )
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = torch.ones_like(response, dtype=attention_mask.dtype, device=attention_mask.device)
        response_attention_mask = response_attention_mask * (response != self.tokenizer.pad_token_id).to(
            response_attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        output = DataProto(batch=batch, non_tensor_batch=dict(prompts.non_tensor_batch))
        output.batch["response_mask"] = _compute_response_mask_from_attention(output)
        elapsed = time.time() - started
        if elapsed > 30 or not getattr(self, "_generate_timing_logged", False):
            logger.warning(
                "BAGEL rollout generate_sequences finished: batch=%s response_length=%s do_sample=%s elapsed=%.2fs",
                batch_size,
                response_length,
                do_sample,
                elapsed,
            )
            self._generate_timing_logged = True
        return output
