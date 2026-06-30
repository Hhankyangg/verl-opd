#!/usr/bin/env python3
"""Convert a verl FSDP BAGEL actor checkpoint to a BAGEL checkpoint directory.

The OPD BAGEL actor checkpoint saves FSDP shards as:

    actor/model_world_size_<N>_rank_<R>.pt

This script merges those shards and writes a BAGEL-style checkpoint by copying
the original BAGEL directory and replacing ``ema.safetensors`` with the merged
actor weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-dir",
        required=True,
        help="Path to global_step_xxx/actor containing model_world_size_*_rank_*.pt.",
    )
    parser.add_argument(
        "--base-bagel-dir",
        required=True,
        help="Original BAGEL checkpoint directory to copy non-trained files from.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output BAGEL checkpoint directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Keep a leading 'model.' prefix in merged keys. Default strips it for BAGEL ema.safetensors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Merge and print key stats, but do not write output files.",
    )
    return parser.parse_args()


def load_world_size(actor_dir: Path) -> int:
    config_path = actor_dir / "fsdp_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    with config_path.open() as f:
        config = json.load(f)
    world_size = config.get("world_size")
    if not world_size:
        raise ValueError(f"world_size not found in {config_path}")
    return int(world_size)


def load_rank_state(actor_dir: Path, world_size: int, rank: int) -> dict[str, Any]:
    shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing {shard_path}")
    return torch.load(shard_path, map_location="cpu", weights_only=False)


def get_mesh_info(rank0_state: dict[str, Any], world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
    first_key = sorted(rank0_state.keys())[0]
    first_value = rank0_state[first_key]
    if isinstance(first_value, DTensor):
        mesh = first_value.device_mesh.mesh
        mesh_dim_names = first_value.device_mesh.mesh_dim_names
    else:
        mesh = np.array([world_size], dtype=np.int64)
        mesh_dim_names = ("fsdp",)
    return mesh, mesh_dim_names


def merge_tensor_shards(name: str, shards: list[Any], mesh_dim_names: tuple[str, ...]) -> torch.Tensor:
    first = shards[0]
    if isinstance(first, DTensor):
        placements = tuple(first.placements)
        if mesh_dim_names and mesh_dim_names[0] in ("dp", "ddp"):
            placements = placements[1:]
        if len(placements) != 1:
            raise NotImplementedError(f"{name}: unsupported DTensor placements={placements}")
        placement = placements[0]
        local_tensors = [shard._local_tensor.bfloat16().cpu() for shard in shards]
        if placement.is_replicate():
            return local_tensors[0].contiguous()
        if placement.is_shard():
            return torch.cat(local_tensors, dim=placement.dim).contiguous()
        raise NotImplementedError(f"{name}: unsupported DTensor placement={placement}")

    tensors = [shard.bfloat16().cpu() if torch.is_tensor(shard) else shard for shard in shards]
    if not torch.is_tensor(tensors[0]):
        raise TypeError(f"{name}: unsupported shard type {type(tensors[0])!r}")
    # Non-DTensor FSDP shards in this codebase are flattened along dim 0.
    return torch.cat(tensors, dim=0).contiguous()


def normalize_key(key: str, keep_prefix: bool) -> str:
    if keep_prefix:
        return key
    for prefix in ("module.model.", "_fsdp_wrapped_module.model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def merge_fsdp_actor(actor_dir: Path, keep_prefix: bool) -> dict[str, torch.Tensor]:
    world_size = load_world_size(actor_dir)
    print(f"Loading rank 0 metadata from world_size={world_size}")
    rank0_state = load_rank_state(actor_dir, world_size, 0)
    mesh, mesh_dim_names = get_mesh_info(rank0_state, world_size)
    print(f"Detected mesh={mesh}, mesh_dim_names={mesh_dim_names}")

    all_states: list[dict[str, Any]] = [rank0_state]
    for rank in tqdm(range(1, world_size), desc="Loading FSDP shards"):
        all_states.append(load_rank_state(actor_dir, world_size, rank))

    keys = sorted(all_states[0].keys())
    merged: dict[str, torch.Tensor] = {}
    for key in tqdm(keys, desc="Merging tensors"):
        shards = []
        for state in all_states:
            if key not in state:
                raise KeyError(f"Missing key {key!r} in one FSDP shard")
            shards.append(state.pop(key))
        out_key = normalize_key(key, keep_prefix=keep_prefix)
        merged[out_key] = merge_tensor_shards(key, shards, mesh_dim_names)

    return merged


def copy_base_checkpoint(base_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    ignore = shutil.ignore_patterns(".git", "*.lock")
    shutil.copytree(base_dir, output_dir, ignore=ignore)


def main() -> None:
    args = parse_args()
    actor_dir = Path(args.actor_dir)
    base_dir = Path(args.base_bagel_dir)
    output_dir = Path(args.output_dir)

    if not actor_dir.is_dir():
        raise NotADirectoryError(actor_dir)
    if not base_dir.is_dir():
        raise NotADirectoryError(base_dir)

    state_dict = merge_fsdp_actor(actor_dir, keep_prefix=args.keep_prefix)
    print(f"Merged {len(state_dict)} tensors")
    preview = list(state_dict.keys())[:10]
    print("First keys:")
    for key in preview:
        print(f"  {key}: shape={tuple(state_dict[key].shape)} dtype={state_dict[key].dtype}")

    if args.dry_run:
        print("Dry run requested; not writing output.")
        return

    copy_base_checkpoint(base_dir, output_dir, overwrite=args.overwrite)
    ema_path = output_dir / "ema.safetensors"
    print(f"Writing merged BAGEL weights to {ema_path}")
    save_file(state_dict, str(ema_path))
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
