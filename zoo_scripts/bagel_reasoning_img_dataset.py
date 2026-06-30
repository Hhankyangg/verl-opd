import copy

import torch

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset.vision_utils import process_image
from verl.utils.model import compute_position_id_with_mask
from zoo_scripts.reasoning_imgvid_dataset import ReasoningImgVidOPDDataset


class BagelReasoningImgDataset(ReasoningImgVidOPDDataset):
    """Image-only BAGEL student dataset.

    This keeps `raw_prompt` and `multi_modal_data["image"]` for BAGEL's own
    understanding encoder while tokenizing a text-only chat prompt for trainer bookkeeping.
    """

    def __getitem__(self, item):
        row_dict: dict = self._get_valid_row_dict(item)
        messages = copy.deepcopy(row_dict[self.prompt_key])
        if row_dict.get(self.video_key):
            raise NotImplementedError("BAGEL student dataset does not support video inputs yet.")

        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config."
            )

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict_images = row_dict.pop(self.image_key, None)
        images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images] if row_dict_images else []
        row_dict["multi_modal_data"] = {"image": images}

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        return row_dict
