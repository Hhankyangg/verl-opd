import copy
import logging
import os

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset


logger = logging.getLogger(__name__)


class ReasoningImgVidOPDDataset(RLHFDataset):
    """Adapt mixed image/video reasoning JSONL records to the OPD RLHF dataset format."""

    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

    @staticmethod
    def _to_file_uri(path: str) -> str:
        return path if "://" in path else f"file://{os.path.abspath(path)}"

    @classmethod
    def _is_video_path(cls, path: str) -> bool:
        return path.lower().endswith(cls.VIDEO_EXTENSIONS)

    @classmethod
    def _normalize_image(cls, image):
        if isinstance(image, str):
            return {"image": cls._to_file_uri(image)}

        if isinstance(image, dict):
            if "bytes" in image:
                return image

            normalized = dict(image)
            image_path = normalized.get("image") or normalized.get("path")
            if isinstance(image_path, str):
                normalized["image"] = cls._to_file_uri(image_path)
                normalized.pop("path", None)
                return normalized

        raise TypeError(f"Unsupported image payload type: {type(image)}")

    @classmethod
    def _normalize_frame_ref(cls, frame):
        if isinstance(frame, str):
            return cls._to_file_uri(frame)

        if isinstance(frame, dict):
            frame_path = frame.get("image") or frame.get("path")
            if not isinstance(frame_path, str):
                raise TypeError(f"Unsupported frame payload: {frame}")
            return cls._to_file_uri(frame_path)

        raise TypeError(f"Unsupported frame payload type: {type(frame)}")

    @classmethod
    def _normalize_video(cls, video):
        if isinstance(video, str):
            return {"video": cls._to_file_uri(video)}

        if isinstance(video, dict):
            normalized = dict(video)

            if "video" in normalized:
                payload = normalized["video"]
                if isinstance(payload, str):
                    normalized["video"] = cls._to_file_uri(payload)
                elif isinstance(payload, list):
                    normalized["video"] = [cls._normalize_frame_ref(frame) for frame in payload]
                else:
                    raise TypeError(f"Unsupported `video` payload type: {type(payload)}")
                return normalized

            if "path" in normalized:
                normalized["video"] = cls._to_file_uri(normalized.pop("path"))
                return normalized

            if "frames" in normalized:
                frames = normalized.pop("frames")
                normalized["video"] = [cls._normalize_frame_ref(frame) for frame in frames]
                return normalized

        if isinstance(video, list):
            if not video:
                raise ValueError("Encountered empty video payload.")
            if len(video) == 1 and isinstance(video[0], str) and cls._is_video_path(video[0]):
                return {"video": cls._to_file_uri(video[0])}
            return {"video": [cls._normalize_frame_ref(frame) for frame in video]}

        raise TypeError(f"Unsupported video payload type: {type(video)}")

    @staticmethod
    def _split_messages(messages):
        prompt_messages = []
        answer = ""

        for idx, message in enumerate(messages):
            if message.get("role") == "assistant":
                answer = message.get("content", "")
                prompt_messages = messages[:idx]
                break
        else:
            prompt_messages = messages

        if not prompt_messages:
            raise ValueError("No prompt messages found after removing assistant turns.")

        return prompt_messages, answer

    @staticmethod
    def _extract_answer(example, assistant_answer):
        if assistant_answer:
            return assistant_answer

        reward_model = example.get("reward_model")
        if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
            return reward_model.get("ground_truth", "")

        extra_info = example.get("extra_info")
        if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
            return extra_info.get("answer", "")

        return ""

    @classmethod
    def _extract_images(cls, example):
        raw_images = example.get("images") or []
        return [cls._normalize_image(image) for image in raw_images]

    @classmethod
    def _extract_videos(cls, example):
        raw_videos = example.get("videos") or []
        raw_video = example.get("video")

        if raw_videos:
            source = raw_videos
        elif raw_video:
            source = [raw_video]
        else:
            return []

        if isinstance(source, list):
            if source and all(isinstance(item, dict) and "video" in item for item in source):
                return [cls._normalize_video(item) for item in source]
            if source and all(isinstance(item, list) for item in source):
                return [cls._normalize_video(item) for item in source]
            if source and all(isinstance(item, str) for item in source):
                if len(source) == 1 and cls._is_video_path(source[0]):
                    return [cls._normalize_video(source[0])]
                return [cls._normalize_video(source)]

        return [cls._normalize_video(source)]

    def _convert_example(self, example, idx):
        messages = example.get("messages") or example.get("prompt") or []
        if not messages:
            raise ValueError("Missing `messages` in reasoning img/video sample.")

        prompt_messages, answer = self._split_messages(messages)
        answer = self._extract_answer(example, answer)
        images = self._extract_images(example)
        videos = self._extract_videos(example)
        user_messages = [message for message in prompt_messages if message.get("role") == "user"]
        raw_question = user_messages[-1]["content"] if user_messages else prompt_messages[-1].get("content", "")
        data_source = example.get("source") or example.get("data_source") or "reasoning"

        return {
            "data_source": data_source,
            "prompt": prompt_messages,
            "images": images,
            "bagel_image_refs": copy.deepcopy(images),
            "videos": videos,
            "ability": example.get("ability", "reasoning"),
            "reward_model": {
                "ground_truth": answer,
                "style": example.get("reward_style", "rule"),
            },
            "extra_info": {
                "answer": answer,
                "bagel_image_refs": copy.deepcopy(images),
                "index": idx,
                "raw_question": raw_question,
                "split": example.get("split", "train"),
                "source": data_source,
                "modality": "video" if videos else "image",
            },
        }

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            if data_file.endswith(".json") or data_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            elif data_file.endswith(".parquet"):
                try:
                    dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
                except TypeError as e:
                    # Some parquet files carry newer HuggingFace feature metadata
                    # that older datasets versions cannot deserialize. Drop the
                    # metadata and let datasets infer columns from the Arrow table.
                    if "dataclass type or instance" not in str(e):
                        raise

                    import pyarrow.parquet as pq

                    logger.warning(
                        "Falling back to pyarrow parquet loading for %s because datasets "
                        "could not parse embedded feature metadata: %s",
                        data_file,
                        e,
                    )
                    table = pq.read_table(data_file).replace_schema_metadata()
                    dataframe = datasets.Dataset(table)
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        self.dataframe = self.dataframe.map(
            self._convert_example,
            with_indices=True,
            remove_columns=self.dataframe.column_names,
            desc="Normalizing mixed reasoning image/video samples for OPD",
        )

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
