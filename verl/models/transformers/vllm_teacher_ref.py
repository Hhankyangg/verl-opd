import base64
import copy
import io
import json
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib import error, request

import torch
from PIL import Image

torch.serialization.add_safe_globals([Image.Image])
torch.serialization.safe_globals([Image.Image])


logger = logging.getLogger(__name__)


class VLLMTeacherRefScorer:
    """Score student text with a remote vLLM OpenAI-compatible teacher."""

    def __init__(
        self,
        base_url: str,
        model: str,
        teacher_tokenizer: Any,
        request_timeout_s: float = 300.0,
        api_key: str | None = None,
        max_concurrent_requests: int = 8,
        max_retries: int = 0,
        retry_backoff_s: float = 30.0,
        restart_url: str | None = None,
        restart_token: str | None = None,
        restart_timeout_s: float = 5.0,
    ):
        if not base_url:
            raise ValueError("vLLM teacher base_url must be set.")
        if not model:
            raise ValueError("vLLM teacher model must be set.")

        self.model = model
        self.teacher_tokenizer = teacher_tokenizer
        self.request_timeout_s = request_timeout_s
        self.api_key = api_key
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.restart_url = restart_url
        self.restart_token = restart_token
        self.restart_timeout_s = max(1.0, float(restart_timeout_s))
        self.chat_completions_url = self._chat_completions_url(base_url)

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    def _request_server_restart(self, reason: str) -> None:
        if not self.restart_url:
            return

        payload = json.dumps({"reason": reason}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.restart_token:
            headers["Authorization"] = f"Bearer {self.restart_token}"
            headers["X-Restart-Token"] = self.restart_token
        restart_request = request.Request(
            self._normalize_url(self.restart_url),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(restart_request, timeout=self.restart_timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
            logger.warning("Requested vLLM teacher restart via %s: %s", self.restart_url, body[:1000])
        except Exception as exc:
            logger.warning("Failed to request vLLM teacher restart via %s: %s", self.restart_url, exc)

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _iter_content_items(cls, content: Any):
        if isinstance(content, str):
            for segment in [item for item in re.split(r"(<image>|<video>)", content) if item != ""]:
                if segment == "<image>":
                    yield {"type": "image"}
                elif segment == "<video>":
                    yield {"type": "video"}
                else:
                    yield {"type": "text", "text": segment}
            return

        for item in content:
            yield item

    @staticmethod
    def _image_content_item(image: Image.Image) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": VLLMTeacherRefScorer._image_to_data_url(image)},
        }

    @classmethod
    def _count_image_placeholders(cls, raw_prompt: list[dict[str, Any]]) -> int:
        count = 0
        for message in raw_prompt:
            content = message.get("content", "")
            count += sum(1 for item in cls._iter_content_items(content) if item.get("type") in {"image", "image_url"})
        return count

    def _messages_for_vllm(
        self,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        images = list((multi_modal_data or {}).get("image", []))
        image_index = 0
        messages = []

        if not getattr(self, "_multimodal_debug_logged", False):
            logger.warning(
                "vLLM teacher multimodal input debug: prompt_image_placeholders=%s multi_modal_images=%s",
                self._count_image_placeholders(raw_prompt),
                len(images),
            )
            self._multimodal_debug_logged = True

        for message in copy.deepcopy(list(raw_prompt)):
            content = message.get("content", "")
            content_items = []
            for item in self._iter_content_items(content):
                item_type = item.get("type")
                if item_type == "text":
                    content_items.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    content_items.append(item)
                elif item_type == "image":
                    if image_index >= len(images):
                        raise ValueError("vLLM teacher prompt referenced more images than multi_modal_data provided.")
                    content_items.append(self._image_content_item(images[image_index]))
                    image_index += 1
                elif item_type == "video":
                    raise NotImplementedError("vLLM teacher ref scorer does not support video inputs.")
                else:
                    raise ValueError(f"Unsupported vLLM teacher content item type: {item_type!r}")

            message["content"] = content_items if any(item["type"] != "text" for item in content_items) else content
            messages.append(message)

        if image_index != len(images):
            raise ValueError(
                f"vLLM teacher prompt consumed {image_index} images but multi_modal_data provided {len(images)}."
            )
        return messages

    def _request_prompt_logprobs(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> tuple[list[int], list[Any]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "add_generation_prompt": add_generation_prompt,
        }
        encoded_payload = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = None
        for attempt in range(self.max_retries + 1):
            http_request = request.Request(
                self.chat_completions_url,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(http_request, timeout=self.request_timeout_s) as response:
                    body = json.load(response)
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"vLLM teacher HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, socket.timeout) as exc:
                self._request_server_restart(
                    f"teacher request timeout after {self.request_timeout_s:.1f}s "
                    f"on attempt {attempt + 1}/{self.max_retries + 1}"
                )
                if attempt < self.max_retries:
                    logger.warning(
                        "vLLM teacher request timed out after %.1fs; retrying attempt %s/%s after %.1fs",
                        self.request_timeout_s,
                        attempt + 1,
                        self.max_retries,
                        self.retry_backoff_s,
                    )
                    if self.retry_backoff_s > 0:
                        time.sleep(self.retry_backoff_s)
                    continue
                raise RuntimeError(
                    "vLLM teacher request timed out after "
                    f"{self.request_timeout_s:.1f}s and {self.max_retries + 1} attempt(s). "
                    "For Qwen2.5-VL-32B prompt_logprobs this usually means the vLLM server is queued/too slow; "
                    "increase VLLM_TEACHER_TIMEOUT_S, reduce TRAIN_BATCH_SIZE, or add more vLLM capacity."
                ) from exc
            except error.URLError as exc:
                raise RuntimeError(f"vLLM teacher request failed: {exc}") from exc

        if body is None:
            raise RuntimeError("vLLM teacher request failed without a response body.")

        response_body = body
        if body.get("choices"):
            # vLLM has moved prompt debug fields between top-level and choice
            # models across OpenAI-compatible response schema revisions.
            response_body = {**body, **body["choices"][0]}
        prompt_token_ids = response_body.get("prompt_token_ids")
        prompt_logprobs = response_body.get("prompt_logprobs")
        if prompt_token_ids is None or prompt_logprobs is None:
            raise RuntimeError(
                "vLLM teacher response did not include prompt_token_ids and prompt_logprobs. "
                "Start a vLLM OpenAI server that supports ChatCompletion prompt_logprobs."
            )
        if len(prompt_token_ids) != len(prompt_logprobs):
            raise RuntimeError(
                "vLLM teacher returned mismatched prompt_token_ids and prompt_logprobs lengths: "
                f"{len(prompt_token_ids)} != {len(prompt_logprobs)}."
            )
        return prompt_token_ids, prompt_logprobs

    @staticmethod
    def _logprob_for_token(token_id: int, token_logprobs: Any) -> float:
        if token_logprobs is None:
            return 0.0
        if not isinstance(token_logprobs, dict):
            raise TypeError(f"Unexpected vLLM prompt logprob entry: {type(token_logprobs).__name__}.")

        logprob = token_logprobs.get(str(token_id), token_logprobs.get(token_id))
        if logprob is None and len(token_logprobs) == 1:
            logprob = next(iter(token_logprobs.values()))
        if isinstance(logprob, dict):
            logprob = logprob.get("logprob")
        if logprob is None:
            raise RuntimeError(f"vLLM prompt logprobs did not contain chosen token id {token_id}.")
        return float(logprob)

    @staticmethod
    def _common_prefix_length(left: list[int], right: list[int]) -> int:
        prefix_length = 0
        for left_token, right_token in zip(left, right, strict=False):
            if left_token != right_token:
                break
            prefix_length += 1
        return prefix_length

    @staticmethod
    def _find_subsequence_from_end(tokens: list[int], pattern: list[int]) -> int:
        if len(pattern) == 0:
            return len(tokens)
        if len(pattern) > len(tokens):
            return -1
        for start in range(len(tokens) - len(pattern), -1, -1):
            if tokens[start : start + len(pattern)] == pattern:
                return start
        return -1

    def _score_response_text(
        self,
        prompt_messages: list[dict[str, Any]],
        response_text: str,
    ) -> list[float]:
        full_messages = copy.deepcopy(prompt_messages)
        full_messages.append({"role": "assistant", "content": response_text})

        full_token_ids, full_prompt_logprobs = self._request_prompt_logprobs(
            full_messages,
            add_generation_prompt=False,
        )
        response_token_ids = self.teacher_tokenizer.encode(response_text, add_special_tokens=False)
        response_start = self._find_subsequence_from_end(full_token_ids, response_token_ids)
        if response_start < 0:
            if not getattr(self, "_alignment_fallback_logged", False):
                logger.warning(
                    "vLLM teacher one-request alignment failed; falling back to prompt/full prefix diff. "
                    "full_tokens=%s response_tokens=%s",
                    len(full_token_ids),
                    len(response_token_ids),
                )
                self._alignment_fallback_logged = True
            prompt_token_ids, _ = self._request_prompt_logprobs(prompt_messages, add_generation_prompt=True)
            response_start = self._common_prefix_length(prompt_token_ids, full_token_ids)
            if response_start == 0 or response_start >= len(full_token_ids):
                raise RuntimeError(
                    "Could not align vLLM teacher prompt_logprobs with either response-token match or prefix diff. "
                    f"prompt_tokens={len(prompt_token_ids)} full_tokens={len(full_token_ids)} "
                    f"response_tokens={len(response_token_ids)}"
                )
            response_token_count = len(full_token_ids) - response_start
        else:
            response_token_count = len(response_token_ids)
        if response_start == len(full_token_ids) or response_token_count == 0:
            return []

        return [
            self._logprob_for_token(token_id, token_logprobs)
            for token_id, token_logprobs in zip(
                full_token_ids[response_start : response_start + response_token_count],
                full_prompt_logprobs[response_start : response_start + response_token_count],
                strict=True,
            )
        ]

    def _score_one(
        self,
        index: int,
        raw_prompt: list[dict[str, Any]],
        sample_multi_modal_data: dict[str, Any],
        response_ids: torch.Tensor,
        valid_response_length: int,
        response_tokenizer: Any,
    ) -> tuple[int, list[float], int]:
        prompt_messages = self._messages_for_vllm(raw_prompt, sample_multi_modal_data)
        response_text = response_tokenizer.decode(response_ids[:valid_response_length])
        if response_tokenizer.eos_token is not None:
            response_text = response_text.replace(response_tokenizer.eos_token, "")
        return index, self._score_response_text(prompt_messages, response_text), valid_response_length

    def score_batch(
        self,
        raw_prompts: Any,
        multi_modal_data_batch: Any,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        response_tokenizer: Any,
    ) -> torch.Tensor:
        score_batch = torch.zeros(responses.shape, dtype=torch.float32)
        response_width = responses.shape[-1]

        jobs = []
        for index, raw_prompt in enumerate(raw_prompts):
            valid_response_length = int(attention_mask[index][-response_width:].sum().item())
            jobs.append(
                (
                    index,
                    list(raw_prompt),
                    multi_modal_data_batch[index] if multi_modal_data_batch is not None else {},
                    responses[index],
                    valid_response_length,
                    response_tokenizer,
                )
            )

        started = time.time()
        concurrency = min(self.max_concurrent_requests, len(jobs) or 1)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(lambda args: self._score_one(*args), jobs))
        elapsed = time.time() - started

        if elapsed > 30 or not getattr(self, "_batch_timing_logged", False):
            logger.warning(
                "vLLM teacher ref scored batch: batch=%s requests=%s concurrency=%s elapsed=%.2fs",
                len(jobs),
                len(jobs),
                concurrency,
                elapsed,
            )
            self._batch_timing_logged = True

        for index, teacher_logprobs, valid_response_length in sorted(results, key=lambda item: item[0]):
            copy_length = min(len(teacher_logprobs), valid_response_length, response_width)
            if copy_length:
                score_batch[index, :copy_length] = torch.tensor(teacher_logprobs[:copy_length], dtype=torch.float32)
            if not getattr(self, "_alignment_debug_logged", False):
                logger.warning(
                    "vLLM teacher ref aligned sample response lengths: bagel_tokens=%s teacher_tokens=%s copied=%s",
                    valid_response_length,
                    len(teacher_logprobs),
                    copy_length,
                )
                self._alignment_debug_logged = True
        return score_batch


class KrakenOPDTeacherRefScorer:
    """Score BAGEL responses with Kraken OPD teacher server `/distill`.

    The server expects Qwen teacher token ids, not BAGEL token ids. This scorer
    rebuilds the full Qwen chat prompt from raw messages plus the BAGEL response,
    sends it to `/distill`, and extracts the teacher logprob of each generated
    response token from the returned top-k distribution.
    """

    def __init__(
        self,
        base_url: str,
        teacher_tokenizer: Any,
        request_timeout_s: float = 300.0,
        max_retries: int = 0,
        retry_backoff_s: float = 30.0,
        restart_url: str | None = None,
        restart_token: str | None = None,
        restart_timeout_s: float = 5.0,
        temperature: float = 1.0,
        missing_token_logprob: float | None = None,
        micro_batch_size: int = 1,
    ):
        if not base_url:
            raise ValueError("Kraken OPD teacher base_url must be set.")

        self.teacher_tokenizer = teacher_tokenizer
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.restart_url = restart_url
        self.restart_token = restart_token
        self.restart_timeout_s = max(1.0, float(restart_timeout_s))
        self.temperature = float(temperature)
        self.missing_token_logprob = missing_token_logprob
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.distill_url = self._distill_url(base_url)

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    @classmethod
    def _distill_url(cls, base_url: str) -> str:
        base_url = cls._normalize_url(base_url)
        if base_url.endswith("/distill"):
            return base_url
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return f"{base_url}/distill"

    @staticmethod
    def _serialize(data: Any) -> bytes:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(raw: bytes) -> Any:
        return torch.load(io.BytesIO(raw), weights_only=False)

    def _request_server_restart(self, reason: str) -> None:
        if not self.restart_url:
            return

        payload = json.dumps({"reason": reason}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.restart_token:
            headers["Authorization"] = f"Bearer {self.restart_token}"
            headers["X-Restart-Token"] = self.restart_token
        restart_request = request.Request(
            self._normalize_url(self.restart_url),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(restart_request, timeout=self.restart_timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
            logger.warning("Requested Kraken OPD teacher restart via %s: %s", self.restart_url, body[:1000])
        except Exception as exc:
            logger.warning("Failed to request Kraken OPD teacher restart via %s: %s", self.restart_url, exc)

    @classmethod
    def _iter_content_items(cls, content: Any):
        if isinstance(content, str):
            for segment in [item for item in re.split(r"(<image>|<video>)", content) if item != ""]:
                if segment == "<image>":
                    yield {"type": "image"}
                elif segment == "<video>":
                    yield {"type": "video"}
                else:
                    yield {"type": "text", "text": segment}
            return

        for item in content:
            yield item

    @staticmethod
    def _image_content_item(image: Image.Image) -> dict[str, Any]:
        return {"type": "image", "image": image.convert("RGB")}

    @classmethod
    def _messages_for_kraken(
        cls,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        images = list((multi_modal_data or {}).get("image", []))
        image_index = 0
        messages = []

        for message in copy.deepcopy(list(raw_prompt)):
            content = message.get("content", "")
            content_items = []
            for item in cls._iter_content_items(content):
                item_type = item.get("type")
                if item_type == "text":
                    content_items.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image":
                    if any(key in item for key in ("image", "image_url", "url", "path")):
                        content_items.append(item)
                    else:
                        if image_index >= len(images):
                            raise ValueError("Kraken OPD teacher prompt referenced more images than provided.")
                        content_items.append(cls._image_content_item(images[image_index]))
                        image_index += 1
                elif item_type == "image_url":
                    content_items.append(item)
                elif item_type == "video":
                    raise NotImplementedError("Kraken OPD teacher ref scorer does not support video inputs.")
                else:
                    raise ValueError(f"Unsupported Kraken OPD teacher content item type: {item_type!r}")
            message["content"] = content_items
            messages.append(message)

        if image_index > 0 and image_index != len(images):
            raise ValueError(
                f"Kraken OPD teacher prompt consumed {image_index} images but multi_modal_data provided {len(images)}."
            )
        return messages

    def _request_distill(
        self,
        input_items: list[tuple[list[int], list[dict[str, Any]], int]],
        debug_summary: str,
    ) -> dict[str, Any]:
        payload = {
            "input_items": input_items,
            "max_tokens": 1,
            "temperature": self.temperature,
            "only_prompt": True,
        }
        encoded_payload = self._serialize(payload)
        headers = {"Content-Type": "application/octet-stream"}

        body = None
        for attempt in range(self.max_retries + 1):
            http_request = request.Request(
                self.distill_url,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(http_request, timeout=self.request_timeout_s) as response:
                    body = self._deserialize(response.read())
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Kraken OPD teacher HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, socket.timeout) as exc:
                self._request_server_restart(
                    f"teacher distill timeout after {self.request_timeout_s:.1f}s "
                    f"on attempt {attempt + 1}/{self.max_retries + 1}"
                )
                if attempt < self.max_retries:
                    logger.warning(
                        "Kraken OPD teacher request timed out after %.1fs; retrying attempt %s/%s after %.1fs",
                        self.request_timeout_s,
                        attempt + 1,
                        self.max_retries,
                        self.retry_backoff_s,
                    )
                    if self.retry_backoff_s > 0:
                        time.sleep(self.retry_backoff_s)
                    continue
                raise RuntimeError(
                    "Kraken OPD teacher request timed out after "
                    f"{self.request_timeout_s:.1f}s and {self.max_retries + 1} attempt(s)."
                ) from exc
            except error.URLError as exc:
                raise RuntimeError(f"Kraken OPD teacher request failed for {self.distill_url}: {exc}") from exc

        if body is None:
            raise RuntimeError("Kraken OPD teacher request failed without a response body.")
        if isinstance(body, dict) and body.get("status") == "error":
            reason = body.get("reason", "unknown")
            if "EngineDeadError" in reason:
                self._request_server_restart(f"Kraken OPD teacher EngineDeadError; {debug_summary}")
            raise RuntimeError(f"Kraken OPD teacher error for {debug_summary}: {reason}")
        return body

    @staticmethod
    def _debug_summary(input_items: list[tuple[list[int], list[dict[str, Any]], int]]) -> str:
        token_lengths = [len(item[0]) for item in input_items]
        response_lengths = [item[2] for item in input_items]
        return (
            f"batch={len(input_items)} "
            f"token_len_min={min(token_lengths, default=0)} token_len_max={max(token_lengths, default=0)} "
            f"response_len_min={min(response_lengths, default=0)} response_len_max={max(response_lengths, default=0)}"
        )

    def _build_item(
        self,
        raw_prompt: list[dict[str, Any]],
        sample_multi_modal_data: dict[str, Any],
        response_ids: torch.Tensor,
        valid_response_length: int,
        response_tokenizer: Any,
    ) -> tuple[tuple[list[int], list[dict[str, Any]], int], list[int]]:
        prompt_messages = self._messages_for_kraken(raw_prompt, sample_multi_modal_data)
        response_text = response_tokenizer.decode(response_ids[:valid_response_length])
        if response_tokenizer.eos_token is not None:
            response_text = response_text.replace(response_tokenizer.eos_token, "")

        full_messages = copy.deepcopy(prompt_messages)
        full_messages.append({"role": "assistant", "content": response_text})
        prompt_token_ids = self.teacher_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_token_ids = self.teacher_tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        teacher_response_length = max(0, len(full_token_ids) - len(prompt_token_ids))
        teacher_response_ids = list(full_token_ids[-teacher_response_length:]) if teacher_response_length > 0 else []
        return (list(full_token_ids), prompt_messages, teacher_response_length), teacher_response_ids

    def _select_response_logprobs(
        self,
        topk_logprobs: torch.Tensor,
        topk_indices: torch.Tensor,
        teacher_response_ids: list[int],
    ) -> tuple[list[float], int]:
        if len(teacher_response_ids) == 0:
            return [], 0

        logprobs = []
        missing = 0
        rows = min(len(teacher_response_ids), topk_logprobs.shape[0], topk_indices.shape[0])
        for token_id, row_logprobs, row_indices in zip(
            teacher_response_ids[:rows],
            topk_logprobs[-rows:],
            topk_indices[-rows:],
            strict=True,
        ):
            matches = row_indices == int(token_id)
            if bool(matches.any()):
                logprobs.append(float(row_logprobs[matches.nonzero(as_tuple=False)[0].item()].item()))
            else:
                missing += 1
                if self.missing_token_logprob is not None:
                    logprobs.append(float(self.missing_token_logprob))
                elif row_logprobs.numel() > 0:
                    logprobs.append(float(row_logprobs.min().item()))
                else:
                    logprobs.append(0.0)
        return logprobs, missing

    def score_batch(
        self,
        raw_prompts: Any,
        multi_modal_data_batch: Any,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        response_tokenizer: Any,
    ) -> torch.Tensor:
        score_batch = torch.zeros(responses.shape, dtype=torch.float32)
        response_width = responses.shape[-1]

        input_items = []
        teacher_response_ids_batch = []
        valid_response_lengths = []
        for index, raw_prompt in enumerate(raw_prompts):
            valid_response_length = int(attention_mask[index][-response_width:].sum().item())
            item, teacher_response_ids = self._build_item(
                list(raw_prompt),
                multi_modal_data_batch[index] if multi_modal_data_batch is not None else {},
                responses[index],
                valid_response_length,
                response_tokenizer,
            )
            input_items.append(item)
            teacher_response_ids_batch.append(teacher_response_ids)
            valid_response_lengths.append(valid_response_length)

        started = time.time()
        topk_logprobs_batch = []
        topk_indices_batch = []
        for start in range(0, len(input_items), self.micro_batch_size):
            end = start + self.micro_batch_size
            chunk = input_items[start:end]
            response = self._request_distill(chunk, self._debug_summary(chunk))
            topk_logprobs_batch.extend(response["teacher_topk_logprobs"])
            topk_indices_batch.extend(response["teacher_topk_indices"])
        elapsed = time.time() - started

        if len(topk_logprobs_batch) != len(input_items) or len(topk_indices_batch) != len(input_items):
            raise RuntimeError(
                "Kraken OPD teacher returned mismatched batch size: "
                f"logprobs={len(topk_logprobs_batch)} indices={len(topk_indices_batch)} expected={len(input_items)}"
            )

        missing_total = 0
        token_total = 0
        for index, (topk_logprobs, topk_indices, teacher_response_ids, valid_response_length) in enumerate(
            zip(topk_logprobs_batch, topk_indices_batch, teacher_response_ids_batch, valid_response_lengths, strict=True)
        ):
            teacher_logprobs, missing = self._select_response_logprobs(
                topk_logprobs,
                topk_indices,
                teacher_response_ids,
            )
            copy_length = min(len(teacher_logprobs), valid_response_length, response_width)
            if copy_length:
                score_batch[index, :copy_length] = torch.tensor(teacher_logprobs[:copy_length], dtype=torch.float32)
            missing_total += missing
            token_total += len(teacher_logprobs)

        if elapsed > 30 or not getattr(self, "_batch_timing_logged", False):
            logger.warning(
                "Kraken OPD teacher scored batch: batch=%s micro_batch=%s elapsed=%.2fs missing_topk=%s/%s",
                len(input_items),
                self.micro_batch_size,
                elapsed,
                missing_total,
                token_total,
            )
            self._batch_timing_logged = True
        return score_batch
