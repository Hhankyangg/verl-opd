import asyncio
import logging
import os
from typing import Any, Optional

import ray

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class BagelRolloutServer:
    """Hybrid BAGEL rollout server.

    This is a lightweight Ray actor that reuses the colocated actor worker
    group's FSDP-sharded BAGEL model. Generation must be invoked on every FSDP
    rank; only rank 0's TokenOutput is returned to the AgentLoop.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ray.actor.ActorHandle],
        replica_rank: int,
    ):
        if rollout_mode != RolloutMode.HYBRID:
            raise NotImplementedError("BAGEL rollout currently supports only hybrid colocated mode.")
        self.config = config
        self.model_config = model_config
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self._generate_lock = asyncio.Lock()

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        raw_prompt: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if video_data:
            raise NotImplementedError("BAGEL rollout does not support video inputs.")
        if raw_prompt is None:
            raise ValueError("BAGEL rollout requires raw_prompt from BagelSingleTurnAgentLoop.")

        async with self._generate_lock:
            futures = [
                worker.bagel_rollout_generate.remote(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    raw_prompt=raw_prompt,
                    **kwargs,
                )
                for worker in self.workers
            ]
            outputs = await asyncio.gather(*futures)
            return outputs[0]

    async def wake_up(self):
        return None

    async def sleep(self):
        return None

    async def abort_all_requests(self):
        return None

    async def resume_generation(self):
        return None

    async def clear_kv_cache(self):
        return None

    async def release_kv_cache(self):
        return None

    async def resume_kv_cache(self):
        return None

    async def start_profile(self, **kwargs):
        return None

    async def stop_profile(self):
        return None


class BagelReplica(RolloutReplica):
    """RolloutReplica adapter for BAGEL student models."""

    async def launch_servers(self):
        if self.rollout_mode != RolloutMode.HYBRID:
            raise NotImplementedError("BAGEL rollout currently supports only hybrid colocated mode.")
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        name = f"bagel_rollout_server_{self.replica_rank}{self.name_suffix}"
        server = BagelRolloutServer.options(name=name, max_concurrency=self.max_concurrency).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
        )
        self.servers = [server]
        self._server_handle = server
        self._server_address = f"bagel://{self.replica_rank}"
