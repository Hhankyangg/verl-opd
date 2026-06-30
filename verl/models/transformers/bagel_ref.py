import copy
import json
import logging
import os
import re
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


logger = logging.getLogger(__name__)


def is_bagel_model_path(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config.get("model_type") == "bagel"


def _ensure_bagel_codebase(codebase_root: str | None = None) -> str:
    root = codebase_root or os.environ.get("BAGEL_CODEBASE")
    if not root:
        raise FileNotFoundError("Set BAGEL_CODEBASE to the local BAGEL repository root.")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"BAGEL codebase directory not found: {root}. "
            "Set BAGEL_CODEBASE to the local BAGEL repository root."
        )

    if root not in sys.path:
        sys.path.insert(0, root)
    return root


class BagelRefScorer:
    """Teacher-side scorer for BAGEL multimodal understanding.

    This scorer rebuilds the teacher prompt from raw chat messages and image
    payloads, then computes token log-probs for the generated response with
    BAGEL's cache-based inference API.
    """

    def __init__(self, model_path: str, device: torch.device | str):
        _ensure_bagel_codebase()

        from safetensors.torch import load_file

        from data.data_utils import add_special_tokens, pil_img2rgb
        from data.transforms import ImageTransform
        from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        from modeling.bagel.qwen2_navit import NaiveCache
        from modeling.qwen2 import Qwen2Tokenizer
        from verl.utils.dataset.vision_utils import process_image

        self._pil_img2rgb = pil_img2rgb
        self._process_image = process_image
        self._naive_cache_cls = NaiveCache
        self.device = torch.device(device)

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=None,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        self.model = Bagel(language_model, vit_model, config).to(device=self.device, dtype=torch.bfloat16)
        self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device=str(self.device))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Missing BAGEL weights: {missing[:20]}")
        # visual generation weights are intentionally skipped
        if unexpected and not all(
            key.startswith(("time_embedder", "vae2llm", "llm2vae", "latent_pos_embed")) for key in unexpected
        ):
            raise RuntimeError(f"Unexpected BAGEL weights: {unexpected[:20]}")
        self.model.eval()
        self.model.requires_grad_(False)

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)

        self.vit_transform = ImageTransform(980, 224, 14)

    def _init_context(self) -> dict[str, Any]:
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": self._naive_cache_cls(self.model.config.llm_config.num_hidden_layers),
        }

    def _update_context_text(self, text: str, gen_context: dict[str, Any]) -> dict[str, Any]:
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = {
            key: value.to(self.device) if torch.is_tensor(value) else value for key, value in generation_input.items()
        }
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    def _update_context_image(self, image: Image.Image, gen_context: dict[str, Any]) -> dict[str, Any]:
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        image = self.vit_transform.resize_transform(self._pil_img2rgb(image))
        generation_input, kv_lens, ropes = self.model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )
        generation_input = {
            key: value.to(self.device) if torch.is_tensor(value) else value for key, value in generation_input.items()
        }
        past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    def _iter_message_segments(
        self,
        raw_prompt: list[dict[str, Any]],
        image_items: list[Image.Image],
    ):
        image_index = 0
        dropped_image_placeholders = 0

        def iter_content_items(content: Any):
            if isinstance(content, str):
                segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment != ""]
                for segment in segments:
                    if segment == "<image>":
                        yield {"type": "image"}
                    elif segment == "<video>":
                        yield {"type": "video"}
                    else:
                        yield {"type": "text", "text": segment}
                return

            for item in content:
                yield item

        for message in raw_prompt:
            role = message.get("role")
            content = message.get("content", "")
            for item in iter_content_items(content):
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text", "")
                    # BAGEL VLM understanding data is packed as interleaved raw text/image
                    # segments rather than chat-template strings with role wrappers.
                    if role in {"user", "human", "system"}:
                        text = text.strip()
                    if text:
                        yield ("text", text)
                elif item_type == "image":
                    if image_index >= len(image_items):
                        dropped_image_placeholders += 1
                        continue
                    yield ("image", image_items[image_index])
                    image_index += 1
                elif item_type == "video":
                    raise NotImplementedError("BAGEL ref adapter does not support video inputs.")

        if dropped_image_placeholders > 0:
            logger.warning(
                "BAGEL ref dropped %s image placeholders because the prompt referenced more images than provided.",
                dropped_image_placeholders,
            )

        if image_index < len(image_items):
            logger.warning(
                "BAGEL ref found %s unused images after rebuilding the prompt; appending them before decoding.",
                len(image_items) - image_index,
            )
            while image_index < len(image_items):
                yield ("image", image_items[image_index])
                image_index += 1

    def _count_image_placeholders(self, raw_prompt: list[dict[str, Any]]) -> int:
        count = 0
        for message in raw_prompt:
            content = message.get("content", "")
            if isinstance(content, str):
                count += content.count("<image>")
            else:
                count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")
        return count

    def _build_context(
        self,
        raw_prompt: list[dict[str, Any]],
        image_items: list[Image.Image],
    ) -> dict[str, Any]:
        gen_context = self._init_context()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for kind, payload in self._iter_message_segments(raw_prompt, image_items):
                if kind == "text":
                    gen_context = self._update_context_text(payload, gen_context)
                else:
                    gen_context = self._update_context_image(payload, gen_context)
        return gen_context

    def _prepare_image_items(self, image_refs: Any) -> list[Image.Image]:
        if image_refs is None:
            return []

        image_items = []
        for image_ref in image_refs:
            if isinstance(image_ref, dict) and "bytes" in image_ref and "image" in image_ref:
                # Some parquet-backed datasets materialize image references with both
                # decoded bytes and a stale image/path field. process_image() expects
                # exactly one source, so prefer bytes which is self-contained.
                image_ref = dict(image_ref)
                image_ref.pop("image", None)
            image_items.append(self._process_image(image_ref))
        return image_items

    def _score_response_ids(self, gen_context: dict[str, Any], response_ids: torch.Tensor) -> torch.Tensor:
        scoring_context = {
            "kv_lens": list(gen_context["kv_lens"]),
            "ropes": list(gen_context["ropes"]),
            "past_key_values": copy.deepcopy(gen_context["past_key_values"]),
        }
        generation_input = self.model.prepare_start_tokens(
            scoring_context["kv_lens"], scoring_context["ropes"], self.new_token_ids
        )
        key_values_lens = generation_input["key_values_lens"].to(self.device)
        packed_key_value_indexes = generation_input["packed_key_value_indexes"].to(self.device)
        packed_query_position_ids = generation_input["packed_query_position_ids"].to(self.device)
        curr_tokens = generation_input["packed_start_tokens"].to(self.device)
        response_ids = response_ids.to(self.device)

        log_probs = []
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for target_token in response_ids:
                packed_text_embedding = self.model.language_model.model.embed_tokens(curr_tokens)
                query_lens = torch.ones_like(curr_tokens)
                packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                    0,
                    len(key_values_lens),
                    device=key_values_lens.device,
                    dtype=key_values_lens.dtype,
                )

                forward_kwargs = {
                    "packed_query_sequence": packed_text_embedding,
                    "query_lens": query_lens,
                    "packed_query_position_ids": packed_query_position_ids,
                    "packed_query_indexes": packed_query_indexes,
                    "past_key_values": scoring_context["past_key_values"],
                    "key_values_lens": key_values_lens,
                    "packed_key_value_indexes": packed_key_value_indexes,
                    "update_past_key_values": True,
                    "is_causal": True,
                }
                if self.model.use_moe:
                    forward_kwargs["mode"] = "und"

                output = self.model.language_model.forward_inference(**forward_kwargs)
                scoring_context["past_key_values"] = output.past_key_values
                logits = self.model.language_model.lm_head(output.packed_query_sequence)
                token_log_prob = F.log_softmax(logits[0], dim=-1)[target_token]
                log_probs.append(token_log_prob)

                curr_tokens = target_token.view(1)
                unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
                for i in range(len(unpacked)):
                    unpacked[i] = torch.cat(
                        [unpacked[i], torch.tensor([unpacked[i][-1] + 1], device=unpacked[i].device)], dim=0
                    )
                packed_key_value_indexes = torch.cat(unpacked, dim=0)
                key_values_lens = key_values_lens + 1
                packed_query_position_ids = packed_query_position_ids + 1

        if not log_probs:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        return torch.stack(log_probs).to(dtype=torch.float32)

    def score_batch(
        self,
        raw_prompts: Any,
        image_refs_batch: Any,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, response_length = responses.shape
        output = torch.zeros((batch_size, response_length), dtype=torch.float32, device=responses.device)

        for i in range(batch_size):
            raw_prompt = raw_prompts[i]
            if not isinstance(raw_prompt, (list, tuple, np.ndarray)):
                raise TypeError(f"raw_prompt must be a list-like chat, got {type(raw_prompt)}")

            image_refs = None
            if image_refs_batch is not None:
                image_refs = image_refs_batch[i]

            image_items = self._prepare_image_items(image_refs)
            image_count = len(image_items)
            placeholder_count = self._count_image_placeholders(list(raw_prompt))
            if placeholder_count != image_count:
                preview = ""
                for message in raw_prompt:
                    content = message.get("content", "")
                    if isinstance(content, str) and content:
                        preview = content[:160]
                        break
                    if isinstance(content, (list, tuple)):
                        text_parts = [
                            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        if text_parts:
                            preview = "".join(text_parts)[:160]
                            break
                logger.warning(
                    "BAGEL ref placeholder mismatch before rebuild: placeholders=%s images=%s preview=%r",
                    placeholder_count,
                    image_count,
                    preview,
                )

            valid_response_length = int(attention_mask[i, -response_length:].sum().item())
            valid_response_ids = responses[i, :valid_response_length]
            if valid_response_length == 0:
                continue

            context = self._build_context(list(raw_prompt), image_items)
            log_probs = self._score_response_ids(context, valid_response_ids)
            output[i, :valid_response_length] = log_probs.to(device=output.device)

        return output
