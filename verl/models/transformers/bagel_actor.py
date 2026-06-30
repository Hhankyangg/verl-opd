import logging
import os
import re
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .bagel_ref import _ensure_bagel_codebase

logger = logging.getLogger(__name__)


class BagelActorModule(nn.Module):
    """Trainable BAGEL understanding module for PPO actor/rollout.

    Evaluation/generation follows BAGEL's native understanding cache path.
    During policy updates, prompt/image context is rebuilt with trainable text/ViT
    cache updates so response loss can flow into the BAGEL understanding stack.
    """

    def __init__(self, model_path: str, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        _ensure_bagel_codebase()

        from safetensors.torch import load_file

        from data.data_utils import add_special_tokens, patchify, pil_img2rgb
        from data.transforms import ImageTransform
        from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        from modeling.bagel.qwen2_navit import NaiveCache
        from modeling.qwen2 import Qwen2Tokenizer

        self._pil_img2rgb = pil_img2rgb
        self._patchify = patchify
        self._naive_cache_cls = NaiveCache
        self.param_dtype = torch_dtype

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        self.model_config = llm_config
        self.config = BagelConfig(
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
        self.model = Bagel(language_model, vit_model, self.config).to(dtype=torch_dtype)
        self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Missing BAGEL weights: {missing[:20]}")
        if unexpected and not all(
            key.startswith(("time_embedder", "vae2llm", "llm2vae", "latent_pos_embed")) for key in unexpected
        ):
            raise RuntimeError(f"Unexpected BAGEL weights: {unexpected[:20]}")

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)
        self.vit_transform = ImageTransform(980, 224, 14)

    def _init_context(self) -> dict[str, Any]:
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": self._naive_cache_cls(self.model.config.llm_config.num_hidden_layers),
        }

    def _count_image_placeholders(self, raw_prompt: list[dict[str, Any]]) -> int:
        count = 0
        for message in raw_prompt:
            content = message.get("content", "")
            if isinstance(content, str):
                count += content.count("<image>")
            else:
                count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")
        return count

    def _iter_message_segments(self, raw_prompt: list[dict[str, Any]], multi_modal_data: dict[str, Any] | None):
        image_items = list((multi_modal_data or {}).get("image", []))
        image_index = 0

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
            yield ("text", f"<|im_start|>{message['role']}\n")
            content = message.get("content", "")
            for item in iter_content_items(content):
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        yield ("text", text)
                elif item_type == "image":
                    if image_index >= len(image_items):
                        raise ValueError("BAGEL actor prompt referenced more images than provided.")
                    yield ("image", image_items[image_index])
                    image_index += 1
                elif item_type == "video":
                    raise NotImplementedError("BAGEL student actor does not support video inputs yet.")
            yield ("text", "<|im_end|>\n")

        if image_index != len(image_items):
            raise ValueError("BAGEL actor found unused images after rebuilding the prompt.")

    @torch.no_grad()
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
    
    def _update_context_text_trainable(self, text: str, gen_context: dict[str, Any]) -> dict[str, Any]:
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
        packed_text_embedding = self.model.language_model.model.embed_tokens(generation_input["packed_text_ids"])
        extra_inputs = {"mode": "und"} if self.model.use_moe else {}
        output = self.model.language_model.model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=generation_input["text_token_lens"],
            packed_query_position_ids=generation_input["packed_text_position_ids"],
            packed_query_indexes=generation_input["packed_text_indexes"],
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            key_values_lens=generation_input["key_values_lens"],
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = output.past_key_values
        return gen_context

    @torch.no_grad()
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
    
    def _update_context_image_trainable(self, image: Image.Image, gen_context: dict[str, Any]) -> dict[str, Any]:
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

        packed_text_embedding = self.model.language_model.model.embed_tokens(generation_input["packed_text_ids"])
        packed_sequence = packed_text_embedding.new_zeros(
            (int(generation_input["packed_seqlens"].sum().item()), self.model.hidden_size)
        )
        packed_sequence[generation_input["packed_text_indexes"]] = packed_text_embedding

        vit_token_seqlens = generation_input["vit_token_seqlens"]
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.model.vit_model(
            packed_pixel_values=generation_input["packed_vit_tokens"],
            packed_flattened_position_ids=generation_input["packed_vit_position_ids"],
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.model.connector(packed_vit_token_embed)
        pos_emb = self.model.vit_pos_embed(generation_input["packed_vit_position_ids"])
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[generation_input["packed_vit_token_indexes"]] = packed_vit_token_embed

        extra_inputs = {"mode": "und"} if self.model.use_moe else {}
        output = self.model.language_model.model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=generation_input["packed_seqlens"],
            packed_query_position_ids=generation_input["packed_position_ids"],
            packed_query_indexes=generation_input["packed_indexes"],
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            key_values_lens=generation_input["key_values_lens"],
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = output.past_key_values
        return gen_context

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _build_context(self, raw_prompt: list[dict[str, Any]], multi_modal_data: dict[str, Any] | None) -> dict[str, Any]:
        gen_context = self._init_context()
        with torch.autocast(device_type=self.device.type, dtype=self.param_dtype):
            for kind, payload in self._iter_message_segments(raw_prompt, multi_modal_data):
                if kind == "text":
                    if self.training:
                        gen_context = self._update_context_text_trainable(payload, gen_context)
                    else:
                        gen_context = self._update_context_text(payload, gen_context)
                else:
                    if self.training:
                        gen_context = self._update_context_image_trainable(payload, gen_context)
                    else:
                        gen_context = self._update_context_image(payload, gen_context)
        return gen_context

    def _build_packed_response_inputs(
        self,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: dict[str, Any] | None,
        response_ids: torch.Tensor,
    ) -> dict[str, Any]:
        packed_text_ids = []
        packed_text_indexes = []
        packed_position_ids = []
        packed_vit_tokens = []
        packed_vit_token_indexes = []
        packed_vit_position_ids = []
        vit_token_seqlens = []
        ce_loss_indexes = []
        packed_label_ids = []
        split_lens = []
        attn_modes = []

        curr_pos = 0
        curr_rope = 0

        def add_text_ids(text_ids: list[int], with_loss: bool = False, labels: list[int] | None = None):
            nonlocal curr_pos, curr_rope
            if not text_ids:
                return
            start = curr_pos
            length = len(text_ids)
            packed_text_ids.extend(text_ids)
            packed_text_indexes.extend(range(start, start + length))
            packed_position_ids.extend(range(curr_rope, curr_rope + length))
            if with_loss:
                if labels is None or len(labels) != length:
                    raise ValueError("BAGEL packed labels must align with response input ids.")
                ce_loss_indexes.extend(range(start, start + length))
                packed_label_ids.extend(labels)
            split_lens.append(length)
            attn_modes.append("causal")
            curr_pos += length
            curr_rope += length

        def add_text(text: str):
            token_ids = self.tokenizer.encode(text)
            add_text_ids([self.new_token_ids["bos_token_id"]] + token_ids + [self.new_token_ids["eos_token_id"]])

        def add_image(image: Image.Image):
            nonlocal curr_pos, curr_rope
            image_tensor = self.vit_transform(self._pil_img2rgb(image))
            vit_tokens = self._patchify(image_tensor, self.model.vit_patch_size)
            num_img_tokens = vit_tokens.shape[0]
            vit_position_ids = self.model.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.model.vit_patch_size,
                max_num_patches_per_side=self.model.vit_max_num_patch_per_side,
            )

            split_len = num_img_tokens + 2
            packed_text_ids.append(self.new_token_ids["start_of_image"])
            packed_text_indexes.append(curr_pos)
            curr_pos += 1

            packed_vit_tokens.append(vit_tokens)
            packed_vit_token_indexes.extend(range(curr_pos, curr_pos + num_img_tokens))
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            curr_pos += num_img_tokens

            packed_text_ids.append(self.new_token_ids["end_of_image"])
            packed_text_indexes.append(curr_pos)
            curr_pos += 1

            packed_position_ids.extend([curr_rope] * split_len)
            split_lens.append(split_len)
            attn_modes.append("full")
            curr_rope += 1

        for kind, payload in self._iter_message_segments(raw_prompt, multi_modal_data):
            if kind == "text":
                add_text(payload)
            else:
                add_image(payload)

        labels = response_ids.to(device=self.device, dtype=torch.long)
        query_ids = torch.cat(
            [
                torch.tensor([self.new_token_ids["bos_token_id"]], dtype=torch.long, device=self.device),
                labels[:-1],
            ],
            dim=0,
        )
        add_text_ids(query_ids.tolist(), with_loss=True, labels=labels.tolist())

        pad_len = (-curr_pos) % 128
        if pad_len:
            add_text_ids([self.new_token_ids["eos_token_id"]] * pad_len)

        packed = {
            "sequence_length": curr_pos,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=self.device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=self.device),
            "sample_lens": [curr_pos],
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=self.device),
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            "ce_loss_indexes": torch.tensor(ce_loss_indexes, dtype=torch.long, device=self.device),
            "packed_label_ids": torch.tensor(packed_label_ids, dtype=torch.long, device=self.device),
            "packed_vit_tokens": None,
            "packed_vit_token_indexes": None,
            "packed_vit_position_ids": None,
            "vit_token_seqlens": None,
        }
        if packed_vit_tokens:
            packed["packed_vit_tokens"] = torch.cat(packed_vit_tokens, dim=0).to(
                device=self.device,
                dtype=self.param_dtype,
            )
            packed["packed_vit_token_indexes"] = torch.tensor(
                packed_vit_token_indexes, dtype=torch.long, device=self.device
            )
            packed["packed_vit_position_ids"] = torch.cat(packed_vit_position_ids, dim=0).to(self.device)
            packed["vit_token_seqlens"] = torch.tensor(vit_token_seqlens, dtype=torch.long, device=self.device)
        return packed

    def _score_response_ids_trainable(
        self,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: dict[str, Any] | None,
        response_ids: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if response_ids.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        if temperature != 1.0:
            raise NotImplementedError("BAGEL packed training logprob path currently requires temperature=1.0.")
        packed = self._build_packed_response_inputs(raw_prompt, multi_modal_data, response_ids)
        was_training = self.model.training
        if not was_training:
            # BAGEL dispatches language-model forward by module.training. The packed
            # logprob path must use the train-style forward even for old-logprob
            # recomputation, where gradients are disabled by the caller.
            self.model.train()
        try:
            output = self.model(**packed)
        finally:
            self.model.train(was_training)
        return -output["ce"].to(torch.float32)

    def _score_response_ids(
        self,
        gen_context: dict[str, Any],
        response_ids: torch.Tensor,
        temperature: float,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if response_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty, empty if calculate_entropy else None

        curr_kvlen = gen_context["kv_lens"][0]
        curr_rope = gen_context["ropes"][0]
        labels = response_ids.to(self.device)
        query_ids = torch.cat(
            [
                torch.tensor([self.new_token_ids["bos_token_id"]], dtype=torch.long, device=self.device),
                labels[:-1],
            ],
            dim=0,
        )
        query_len = labels.numel()
        packed_query_position_ids = torch.arange(curr_rope, curr_rope + query_len, device=self.device, dtype=torch.long)
        packed_query_indexes = torch.arange(curr_kvlen, curr_kvlen + query_len, device=self.device, dtype=torch.long)
        packed_key_value_indexes = torch.arange(curr_kvlen, device=self.device, dtype=torch.long)
        key_values_lens = torch.tensor([curr_kvlen], device=self.device, dtype=torch.int)
        query_lens = torch.tensor([query_len], device=self.device, dtype=torch.int)

        packed_query_sequence = self.model.language_model.model.embed_tokens(query_ids)
        extra_inputs = {"mode": "und"} if self.model.use_moe else {}
        output = self.model.language_model.model.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=gen_context["past_key_values"],
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=True,
            **extra_inputs,
        )
        logits = self.model.language_model.lm_head(output.packed_query_sequence)
        logits = logits.to(torch.float32)
        logits.div_(temperature)
        log_probs = F.log_softmax(logits, dim=-1)[torch.arange(query_len, device=self.device), labels]
        entropy = None
        if calculate_entropy:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return log_probs, entropy

    def _generate_response_ids(
        self,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: dict[str, Any] | None,
        max_length: int,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        gen_context = self._build_context(raw_prompt, multi_modal_data)
        generation_input = self.model.prepare_start_tokens(gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids)
        generation_input = {
            key: value.to(self.device) if torch.is_tensor(value) else value for key, value in generation_input.items()
        }
        generated = self._generate_text_synced(
            past_key_values=gen_context["past_key_values"],
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            key_values_lens=generation_input["key_values_lens"],
            packed_start_tokens=generation_input["packed_start_tokens"],
            packed_query_position_ids=generation_input["packed_query_position_ids"],
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
        )
        if generated.shape[0] <= 1:
            return torch.empty(0, dtype=torch.long, device=self.device)
        response_ids = generated[1:, 0]
        eos_positions = torch.nonzero(response_ids == self.new_token_ids["eos_token_id"], as_tuple=False)
        if eos_positions.numel() > 0:
            response_ids = response_ids[: int(eos_positions[0].item())]
        return response_ids

    @torch.no_grad()
    def _generate_text_synced(
        self,
        past_key_values,
        packed_key_value_indexes: torch.Tensor,
        key_values_lens: torch.Tensor,
        packed_start_tokens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        max_length: int,
        do_sample: bool,
        temperature: float,
        end_token_id: int,
    ) -> torch.Tensor:
        generated_sequence = []
        curr_tokens = packed_start_tokens
        local_finished = torch.zeros((), dtype=torch.bool, device=self.device)
        eos_tokens = torch.full_like(curr_tokens, end_token_id)

        for _ in range(max_length):
            curr_tokens = torch.where(local_finished.expand_as(curr_tokens), eos_tokens, curr_tokens)
            generated_sequence.append(curr_tokens)

            packed_text_embedding = self.model.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0,
                len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype,
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {"mode": "und"} if self.model.use_moe else {}
            output = self.model.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            pred_logits = self.model.language_model.lm_head(output.packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(pred_logits, dim=-1)

            if dist.is_available() and dist.is_initialized():
                dist.broadcast(next_tokens, src=0)

            local_finished = local_finished | (next_tokens == end_token_id).all()
            curr_tokens = torch.where(local_finished.expand_as(next_tokens), eos_tokens, next_tokens)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1

            if self._all_generation_ranks_finished(local_finished):
                break

        if len(generated_sequence) < max_length and not getattr(self, "_sync_early_stop_logged", False):
            logger.warning(
                "BAGEL synchronized generation early-stopped after %s/%s decoder steps.",
                len(generated_sequence),
                max_length,
            )
            self._sync_early_stop_logged = True

        output_device = generated_sequence[0].device
        return torch.stack([tokens.to(output_device) for tokens in generated_sequence], dim=0)

    def _all_generation_ranks_finished(self, local_finished: torch.Tensor) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return bool(local_finished.item())

        finished_count = local_finished.to(dtype=torch.int32)
        dist.all_reduce(finished_count, op=dist.ReduceOp.SUM)
        return int(finished_count.item()) == dist.get_world_size()

    def forward(
        self,
        mode: str,
        raw_prompts: Any,
        multi_modal_data_batch: Any | None = None,
        responses: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        calculate_entropy: bool = False,
        max_length: int | None = None,
        do_sample: bool = True,
    ):
        if mode == "log_prob":
            assert responses is not None and attention_mask is not None
            batch_size, response_length = responses.shape
            output = torch.zeros((batch_size, response_length), dtype=torch.float32, device=responses.device)
            entropy_output = None
            if calculate_entropy:
                entropy_output = torch.zeros_like(output)

            for i in range(batch_size):
                raw_prompt = raw_prompts[i]
                multi_modal_data = multi_modal_data_batch[i] if multi_modal_data_batch is not None else None
                valid_response_length = int(attention_mask[i, -response_length:].sum().item())
                valid_response_ids = responses[i, :valid_response_length]
                if valid_response_length == 0:
                    continue

                if self.training or not calculate_entropy:
                    if calculate_entropy:
                        raise NotImplementedError("BAGEL packed training logprob path does not support entropy yet.")
                    log_probs = self._score_response_ids_trainable(
                        list(raw_prompt),
                        multi_modal_data,
                        valid_response_ids,
                        temperature=temperature,
                    )
                    entropy = None
                else:
                    context = self._build_context(list(raw_prompt), multi_modal_data)
                    log_probs, entropy = self._score_response_ids(
                        context, valid_response_ids, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                output[i, :valid_response_length] = log_probs.to(device=output.device)
                if calculate_entropy and entropy_output is not None:
                    entropy_output[i, :valid_response_length] = entropy.to(device=entropy_output.device)

            return entropy_output, output

        if mode == "generate":
            assert max_length is not None
            responses_out = []
            for i, raw_prompt in enumerate(raw_prompts):
                multi_modal_data = multi_modal_data_batch[i] if multi_modal_data_batch is not None else None
                response_ids = self._generate_response_ids(
                    list(raw_prompt),
                    multi_modal_data,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                responses_out.append(response_ids)
            return responses_out

        raise ValueError(f"Unsupported BAGEL actor mode: {mode}")
