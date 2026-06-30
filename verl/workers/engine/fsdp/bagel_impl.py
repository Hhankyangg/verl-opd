import logging
import os
from contextlib import nullcontext
from typing import ContextManager

import torch
from tensordict import TensorDict

from verl.models.transformers.bagel_actor import BagelActorModule
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.vision_utils import process_image
from verl.utils.device import get_device_id, get_device_name
from verl.utils.torch_dtypes import PrecisionType

from ..base import EngineRegistry
from .transformer_impl import FSDPEngine

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="bagel", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithBagel(FSDPEngine):
    """FSDP engine for BAGEL understanding actor.

    BAGEL is not a HuggingFace AutoModel, so the standard LM-head engine cannot
    construct or call it. This engine keeps verl's optimizer/FSDP/checkpoint
    plumbing, but delegates response log-prob computation to BagelActorModule.
    """

    def _build_module(self):
        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        module = BagelActorModule(model_path=self.model_config.local_path, torch_dtype=torch_dtype)
        if self.model_config.enable_gradient_checkpointing:
            logger.warning("BAGEL gradient checkpointing is not wired in the new engine; continuing without it.")
        return module

    def _unwrap_non_tensor_batch(self, value):
        if value is None:
            return None
        value = tu.unwrap_non_tensor_data(value)
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        try:
            return [tu.unwrap_non_tensor_data(item) for item in value]
        except TypeError:
            return value

    def _response_scores_to_sequence_nested(self, scores: torch.Tensor, data: TensorDict) -> torch.Tensor:
        attention_mask = data["attention_mask"]
        prompt_width = data["prompts"].shape[-1]
        response_width = data["responses"].shape[-1]
        prompt_lens = attention_mask[:, :prompt_width].sum(dim=1).to(torch.long)
        response_lens = attention_mask[:, prompt_width : prompt_width + response_width].sum(dim=1).to(torch.long)

        sequence_scores = []
        for i, (prompt_len, response_len) in enumerate(zip(prompt_lens.tolist(), response_lens.tolist(), strict=True)):
            if prompt_len <= 0:
                raise ValueError("BAGEL scoring requires non-empty prompts.")
            seq_score = torch.zeros(prompt_len + response_len, dtype=scores.dtype, device=scores.device)
            if response_len > 0:
                seq_score[prompt_len - 1 : prompt_len - 1 + response_len] = scores[i, :response_len]
            sequence_scores.append(seq_score)
        return torch.nested.as_nested_tensor(sequence_scores, layout=torch.jagged)

    def _forward_bagel_log_probs(self, micro_batch: TensorDict, calculate_entropy: bool) -> dict[str, torch.Tensor]:
        raw_prompts = self._unwrap_non_tensor_batch(micro_batch.get("raw_prompt", None))
        if raw_prompts is None:
            raise ValueError("BAGEL actor requires raw_prompt. Set data.return_raw_chat=True.")

        multi_modal_data_batch = self._unwrap_non_tensor_batch(micro_batch.get("multi_modal_data", None))
        if multi_modal_data_batch is None:
            image_refs_batch = self._unwrap_non_tensor_batch(micro_batch.get("bagel_image_refs", None))
            if image_refs_batch is not None:
                rebuilt_batch = []
                for image_refs in image_refs_batch:
                    images = []
                    for image_ref in image_refs or []:
                        if isinstance(image_ref, dict) and "bytes" in image_ref and "image" in image_ref:
                            image_ref = dict(image_ref)
                            image_ref.pop("image", None)
                        images.append(process_image(image_ref))
                    rebuilt_batch.append({"image": images})
                multi_modal_data_batch = rebuilt_batch
        temperature = tu.get_non_tensor_data(data=micro_batch, key="temperature", default=1.0)
        if isinstance(temperature, torch.Tensor):
            temperature = float(temperature.item())

        _, response_log_probs = self.module(
            mode="log_prob",
            raw_prompts=raw_prompts,
            multi_modal_data_batch=multi_modal_data_batch,
            responses=micro_batch["responses"],
            attention_mask=micro_batch["attention_mask"],
            temperature=temperature,
            calculate_entropy=False,
        )
        log_probs = self._response_scores_to_sequence_nested(response_log_probs, micro_batch)
        model_output = {"log_probs": log_probs}
        if calculate_entropy:
            model_output["entropy"] = self._response_scores_to_sequence_nested(
                torch.zeros_like(response_log_probs), micro_batch
            )
        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        autocast_dtype = getattr(self, "_autocast_dtype", torch.bfloat16)
        autocast_ctx: ContextManager = (
            nullcontext()
            if autocast_dtype == torch.float32
            else torch.autocast(device_type=device_name, dtype=autocast_dtype)
        )
        with autocast_ctx:
            model_output = self._forward_bagel_log_probs(micro_batch, calculate_entropy=calculate_entropy)

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            detached_output = {
                key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
                for key, value in model_output.items()
            }
            return loss, {"model_output": detached_output, "loss": loss.detach().item(), "metrics": metrics}
