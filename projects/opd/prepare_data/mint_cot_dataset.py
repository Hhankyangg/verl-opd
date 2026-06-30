import argparse
import glob
import os
import re

import datasets


def _normalize_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question)}")
    return question.strip()


def _normalize_images(example: dict) -> list:
    image_value = example.get("new_images")
    if image_value is None:
        image_value = example.get("images")
    if image_value is None:
        image_value = example.get("image")

    if image_value is None:
        return []
    if isinstance(image_value, list):
        return image_value
    return [image_value]


def _extract_final_answer(solution: str) -> str:
    if not isinstance(solution, str):
        raise TypeError(f"solution must be a string, got {type(solution)}")

    marker = "### The final answer is:"
    suffix = solution.rsplit(marker, 1)[-1].strip() if marker in solution else solution.strip()

    for line in suffix.splitlines():
        cleaned = line.strip()
        if cleaned:
            return re.sub(r"[.。]+$", "", cleaned).strip()
    return suffix


def _load_train_split(local_dataset_path: str | None, dataset_path: str):
    if local_dataset_path is None:
        dataset = datasets.load_dataset(dataset_path)
        return dataset["train"]

    expanded_path = os.path.expanduser(local_dataset_path)
    if os.path.isfile(expanded_path) and expanded_path.endswith(".parquet"):
        dataset = datasets.load_dataset("parquet", data_files={"train": expanded_path})
        return dataset["train"]

    if os.path.isdir(expanded_path):
        parquet_patterns = [
            os.path.join(expanded_path, "train*.parquet"),
            os.path.join(expanded_path, "rl", "*.parquet"),
            os.path.join(expanded_path, "**", "*.parquet"),
        ]
        parquet_files = []
        for pattern in parquet_patterns:
            parquet_files.extend(glob.glob(pattern, recursive=True))
        parquet_files = sorted(set(parquet_files))
        if parquet_files:
            dataset = datasets.load_dataset("parquet", data_files={"train": parquet_files})
            return dataset["train"]

    dataset = datasets.load_dataset(expanded_path)
    return dataset["train"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Optional local dataset path or parquet file.")
    parser.add_argument(
        "--local_save_dir",
        default="projects/opd/data_parquet/mint_cot_dataset",
        help="Directory to save processed parquet files.",
    )
    parser.add_argument("--val_size", type=int, default=100, help="Number of examples to save as val_sample100.parquet.")
    args = parser.parse_args()

    dataset_path = "xy06/MINT-CoT-Dataset"
    train_dataset = _load_train_split(args.local_dataset_path, dataset_path)

    def process_fn(example, idx):
        question = _normalize_question(example.get("question", example.get("problem")))
        response = example.get("response", example.get("solution", ""))
        final_answer = _extract_final_answer(response)
        images = _normalize_images(example)
        has_image = len(images) > 0

        prompt_text = question
        if has_image and "<image>" not in prompt_text.lower():
            prompt_text = "<image>\n" + prompt_text

        data = {
            "data_source": "xy06/MINT-CoT-Dataset",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": final_answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "raw_question": example.get("question", example.get("problem")),
                "answer": final_answer,
            },
        }
        if has_image:
            data["images"] = images
        return data

    train_dataset = train_dataset.map(
        function=process_fn,
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "val_sample100.parquet")
    train_dataset.to_parquet(train_path)
    train_dataset.select(range(min(args.val_size, len(train_dataset)))).to_parquet(val_path)

    print(f"saved train parquet to {train_path}")
    print(f"saved val parquet to {val_path}")
    print(train_dataset[0])


if __name__ == "__main__":
    main()
