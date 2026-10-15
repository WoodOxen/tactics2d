# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""SMART network and its behavior-model entry point."""

import dataclasses
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import Trajectory

from .agent_decoder import SmartAgentDecoder
from .batch_builder import SmartBatchBuilder, observed_frames
from .codebook import load_codebook
from .config import SmartConfig
from .map_decoder import SmartMapDecoder
from .policy import TorchSmartPolicy
from .prediction import SmartPrediction
from .rollout import SmartRolloutResult, SmartRolloutRunner
from .tokens import SmartTokenBatch


class SmartTorchModel(nn.Module):
    """The SMART encoder stack, laid out to consume a released checkpoint.

    Submodule names must match the checkpoint's ``encoder.map_encoder.*`` and
    ``encoder.agent_encoder.*`` keys.

    Attributes:
        map_encoder (SmartMapDecoder): Map encoder.
        agent_encoder (SmartAgentDecoder): Agent decoder and rollout.
    """

    def __init__(self, config: Optional[SmartConfig] = None):
        """Initialize the stack and its codebooks.

        Args:
            config (Optional[SmartConfig], optional): Port configuration. Defaults to None.

        Raises:
            FileNotFoundError: If a codebook cannot be found.
        """

        super().__init__()
        self.config = config if config is not None else SmartConfig()
        motion_codebook = load_codebook(self.config.motion_codebook, "motion codebook")
        map_codebook = load_codebook(self.config.map_codebook, "map codebook")
        map_token = {"traj_src": torch.from_numpy(map_codebook["traj_src"]).to(torch.float)}
        self.map_encoder = SmartMapDecoder(self.config, map_token)
        self.agent_encoder = SmartAgentDecoder(self.config, motion_codebook)

    def inference(self, batch, future_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Encode the scene and roll every modeled agent forward.

        Args:
            batch: A token batch carrying ``agents`` and ``map_tokens``.

        Returns:
            The agent decoder's output dict.
        """

        map_enc = self.encode_map(batch)
        return self.agent_encoder.inference(
            batch.agents, batch.map_tokens, map_enc, future_steps=future_steps
        )

    def encode_map(self, batch) -> Dict[str, torch.Tensor]:
        """Encode the static map once for one or more stochastic rollouts."""

        return self.map_encoder(batch.map_tokens)

    def sample_futures(
        self, batch, seeds: List[int], map_enc: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Decode several independent samples in one network micro-batch."""

        if not seeds:
            raise ValueError("at least one sample seed is required")
        encoded = self.encode_map(batch) if map_enc is None else map_enc
        device = encoded["x_pt"].device
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        output = self.agent_encoder.inference(
            batch.agents, batch.map_tokens, encoded, sample_count=len(seeds), generators=generators
        )
        agents = batch.agents.num_agents
        for name in ("pred_traj", "pred_head", "pred_prob", "next_token_idx"):
            output[name] = output[name].reshape(len(seeds), agents, *output[name].shape[1:])
        return output

    @staticmethod
    def encoder_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Strip the checkpoint's Lightning module prefix from every key.

        Args:
            state_dict (Dict[str, torch.Tensor]): Raw checkpoint state dict.

        Returns:
            A dict keyed by this module's own parameter names.

        Raises:
            ValueError: If a key does not carry the expected prefix.
        """

        prefix = "encoder."
        stripped = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                raise ValueError(
                    f"checkpoint key {key!r} does not start with {prefix!r}; this loader "
                    "expects the released SMART encoder checkpoint."
                )
            stripped[key[len(prefix) :]] = value
        return stripped


def _state_dict_from_checkpoint(payload) -> Dict[str, torch.Tensor]:
    """Return the weights inside a Lightning checkpoint envelope.

    Args:
        payload: Either the checkpoint dict or a bare state dict.

    Returns:
        The weights, keyed as the checkpoint stored them.

    Raises:
        ValueError: If *payload* carries no recognisable state dict.
    """

    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError(
        "checkpoint carries no state dict: expected a 'state_dict' or "
        "'model_state_dict' entry, or the weights themselves."
    )


class SmartBehaviorModel(BehaviorModelBase):
    """Public SMART behavior model entry point.

    SMART decodes the scene **jointly**: one forward pass produces every
    modeled agent's future at once, and ``predict`` only filters what it returns.
    """

    def __init__(
        self,
        config: Optional[SmartConfig] = None,
        policy=None,
        builder: Optional[SmartBatchBuilder] = None,
        device=None,
        dtype=None,
    ):
        """Initialize the model.

        Args:
            config (Optional[SmartConfig], optional): Port configuration. Defaults to None.
            policy (object, optional): Object with a ``predict_batch`` method. Defaults to
                None, which raises.
            builder (Optional[SmartBatchBuilder], optional): Batch builder. Defaults to
                None, which builds one from *config*.
            device (Optional[str], optional): Device name. Defaults to None.
            dtype (Optional[torch.dtype], optional): Parameter dtype. Defaults to None.

        Raises:
            ValueError: If *policy* is None.
        """

        if policy is None:
            raise ValueError(
                "A policy is required. Pass a TorchSmartPolicy, or any object "
                "with predict_batch(), or use from_checkpoint() to construct a "
                "model from weights."
            )
        self.config = config if config is not None else SmartConfig()
        self.policy = policy
        self.builder = builder if builder is not None else SmartBatchBuilder(self.config)
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        config: Optional[SmartConfig] = None,
        map_location=None,
        device=None,
        dtype=None,
    ) -> "SmartBehaviorModel":
        """Load a released SMART checkpoint into a ready-to-use model.

        The load is strict: a missing or unexpected key means the port has drifted.

        Args:
            path (str): Path to a ``.ckpt`` file or a bare state dict.
            config (Optional[SmartConfig], optional): Configuration to build with. Defaults to None.
            map_location (optional): ``torch.load`` argument. Defaults to CPU; the
                constructed module is moved to *device* after loading.
            device (Optional[str], optional): Device to place the model on. Defaults to None.
            dtype (Optional[torch.dtype], optional): Parameter dtype. Defaults to None.

        Returns:
            The loaded model, in eval mode, seeded from ``config.seed``.

        Raises:
            RuntimeError: If the state dict does not match the port's modules.
        """

        if map_location is None:
            # Deserialize on CPU first, matching the released SMART loader.  In
            # SMART's PyTorch 1.12 environment, restoring CUDA storages directly
            # from this archive can crash inside torch.load even though ordinary
            # CUDA allocation and moving the completed module both work.
            map_location = "cpu"
        payload = torch.load(path, map_location=map_location)
        resolved = config if config is not None else SmartConfig()

        module = SmartTorchModel(resolved)
        module.load_state_dict(
            SmartTorchModel.encoder_state_dict(_state_dict_from_checkpoint(payload))
        )
        module.eval()
        if device is not None:
            module.to(device)
        if dtype is not None:
            module.to(dtype=dtype)

        policy = TorchSmartPolicy(module, device=device, dtype=dtype, seed=resolved.seed)
        return cls(
            config=resolved,
            policy=policy,
            builder=SmartBatchBuilder(resolved),
            device=device,
            dtype=dtype,
        )

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Predict the futures of the requested agents.

        The scene is decoded jointly however few agents are asked for; only the
        returned dict is filtered.

        Args:
            participants (Dict[object, object]): All participants, keyed by id.
            map_ (Optional[Map]): The map to tokenize.
            frame (int): Newest observed frame of the history window, in milliseconds.
            agent_ids (Optional[Iterable[object]], optional): Agents whose trajectories
                are wanted. Defaults to None, which returns every modeled agent.

        Returns:
            One trajectory per requested agent the scene modeled, keyed by id.
        """

        wanted = None if agent_ids is None else list(agent_ids)
        prediction = self.predict_scene(participants, map_, frame, agent_ids=wanted)
        if wanted is None:
            wanted = list(prediction.agent_ids)
        return {
            agent_id: prediction.trajectory(agent_id)
            for agent_id in wanted
            if agent_id in prediction.agent_ids
        }

    def predict_scene(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        future_steps: Optional[int] = None,
    ) -> SmartPrediction:
        """Decode the scene jointly and return every modeled agent's rollout.

        Args:
            participants (Dict[object, object]): All participants, keyed by id.
            map_ (Optional[Map]): The map to tokenize.
            frame (int): Newest observed frame of the history window, in milliseconds.
            agent_ids (Optional[Iterable[object]], optional): Agents to center the modeled
                set on; the selection radius is measured from the first id. Defaults to None.

        Returns:
            The joint rollout, stamped with the scenario's own frame timestamps.

        Raises:
            ValueError: If *map_* is None, or if no modeled participant is active at *frame*.
        """

        batch = self.prepare_scene(participants, map_, frame, agent_ids=agent_ids)
        return self.predict_prepared_scene(batch, participants, future_steps=future_steps)

    def prepare_scene(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> SmartTokenBatch:
        """Tokenize one scene once for one or more independent samples.

        A prepared batch contains no rollout state and can therefore be reused
        safely when a caller needs several stochastic futures for the same
        observed scene.
        """

        if map_ is None:
            raise ValueError("SMART needs a map to tokenize; map_ is None.")
        center_id = None if agent_ids is None else next(iter(agent_ids), None)
        batch = self.builder.build(participants, map_, frame, center_id=center_id)
        return batch if self.device is None else batch.to(self.device)

    def predict_prepared_scene(
        self,
        batch: SmartTokenBatch,
        participants: Dict[object, object],
        map_encoding=None,
        future_steps: Optional[int] = None,
    ) -> SmartPrediction:
        """Sample a joint future from an already tokenized scene batch."""

        prediction = self.policy.predict_batch(
            batch, map_encoding=map_encoding, future_steps=future_steps
        )
        return dataclasses.replace(
            prediction,
            frames=self._future_frames(
                participants, batch, future_steps=prediction.positions.shape[1]
            ),
        )

    def encode_prepared_map(self, batch: SmartTokenBatch):
        """Encode a prepared scene's static map for reuse across samples."""

        return self.policy.encode_map(batch)

    def predict_prepared_samples(
        self,
        batch: SmartTokenBatch,
        participants: Dict[object, object],
        seeds: List[int],
        map_encoding=None,
    ) -> List[SmartPrediction]:
        """Return independent stochastic rollouts in one micro-batch."""

        frames = self._future_frames(participants, batch)
        return [
            dataclasses.replace(prediction, frames=frames)
            for prediction in self.policy.predict_samples(batch, seeds, map_encoding=map_encoding)
        ]

    def _future_frames(
        self,
        participants: Dict[object, object],
        batch: SmartTokenBatch,
        future_steps: Optional[int] = None,
    ) -> List[int]:
        """Return the scenario's own timestamps for the predicted steps.

        Args:
            participants (Dict[object, object]): All participants, keyed by id.
            batch (SmartTokenBatch): The batch the rollout came from.

        Returns:
            One timestamp per predicted step, on the scenario's grid then the lattice.
        """

        anchor = int(batch.frame_ms)
        step = self.config.step_ms
        count = self.config.future_steps if future_steps is None else int(future_steps)
        ahead = [frame for frame in observed_frames(participants) if frame > anchor][:count]
        if len(ahead) == count:
            return ahead
        return ahead + [anchor + step * (index + 1) for index in range(len(ahead), count)]

    def rollout(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        start_frame: Optional[int] = None,
        controlled_ids: Optional[Iterable[object]] = None,
        *,
        warmup_steps: Optional[int] = None,
        planning_interval: int = 10,
        scenario_steps: Optional[int] = None,
    ) -> SmartRolloutResult:
        """Replay a scenario closed-loop and return its outcome.

        Args:
            participants (Dict[object, object]): All participants in the scenario.
            map_ (Optional[Map]): The map to tokenize.
            ego_id (object): The agent the loop is centered on.
            start_frame (Optional[int], optional): Timestamp of index 0, in
                milliseconds. Defaults to None, which uses the scenario's own first
                observed frame.
            warmup_steps (Optional[int], optional): Steps of ground truth before the first
                replan. Defaults to None, which uses the history length.
            planning_interval (int, optional): Steps between replans. Defaults to 10.
            scenario_steps (Optional[int], optional): Number of scenario steps. Defaults to
                None, which uses the model's token span.
            controlled_ids (Optional[Iterable], optional): The agents to commit each
                cycle. Defaults to None, which commits every agent the decoder
                returns. When given, *ego_id* must be a member, and only agents the
                decoder also modeled are committed - see ``SmartRolloutRunner.run``.

        Returns:
            The closed-loop outcome.
        """

        runner = SmartRolloutRunner(
            self,
            config=self.config,
            warmup_steps=warmup_steps,
            planning_interval=planning_interval,
            scenario_steps=scenario_steps,
        )
        return runner.run(
            participants, map_, ego_id, start_frame=start_frame, controlled_ids=controlled_ids
        )
