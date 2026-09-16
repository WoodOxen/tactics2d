# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""SMART network and its behavior-model entry point."""

import dataclasses
import pickle
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import Trajectory

from .agent_decoder import SmartAgentDecoder
from .config import SmartConfig
from .dataset import SmartBatchBuilder, _observed_frames
from .map_decoder import SmartMapDecoder
from .policy import TorchSmartPolicy
from .rolling import SmartRollingRunner
from .schema import SmartPrediction, SmartRollingResult, SmartTokenBatch


class SmartTorchModel(nn.Module):
    """The SMART encoder stack, laid out to consume an upstream checkpoint.

    Submodule names mirror the checkpoint's own ``encoder.map_encoder.*`` and
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
        with open(self.config.motion_codebook, "rb") as handle:
            motion_codebook = pickle.load(handle)
        with open(self.config.map_codebook, "rb") as handle:
            map_codebook = pickle.load(handle)
        map_token = {"traj_src": torch.from_numpy(map_codebook["traj_src"]).to(torch.float)}
        self.map_encoder = SmartMapDecoder(self.config, map_token)
        self.agent_encoder = SmartAgentDecoder(self.config, motion_codebook)

    def inference(self, batch) -> Dict[str, torch.Tensor]:
        """Encode the scene and roll every modelled agent forward.

        Args:
            batch: A token batch carrying ``agents`` and ``map_tokens``.

        Returns:
            The agent decoder's output dict.
        """

        map_enc = self.map_encoder(batch.map_tokens)
        return self.agent_encoder.inference(batch.agents, batch.map_tokens, map_enc)

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
    modelled agent's future at once, and ``predict`` only filters what it returns.
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
            map_location (optional): ``torch.load`` argument. Defaults to *device*, else CPU.
            device (Optional[str], optional): Device to place the model on. Defaults to None.
            dtype (Optional[torch.dtype], optional): Parameter dtype. Defaults to None.

        Returns:
            The loaded model, in eval mode, seeded from ``config.seed``.

        Raises:
            RuntimeError: If the state dict does not match the port's modules.
        """

        if map_location is None:
            map_location = device if device is not None else "cpu"
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
                are wanted. Defaults to None, which returns every modelled agent.

        Returns:
            One trajectory per requested agent the scene modelled, keyed by id.
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
    ) -> SmartPrediction:
        """Decode the scene jointly and return every modelled agent's rollout.

        Args:
            participants (Dict[object, object]): All participants, keyed by id.
            map_ (Optional[Map]): The map to tokenize.
            frame (int): Newest observed frame of the history window, in milliseconds.
            agent_ids (Optional[Iterable[object]], optional): Agents to centre the modelled
                set on; the selection radius is measured from the first id. Defaults to None.

        Returns:
            The joint rollout, stamped with the scenario's own frame timestamps.

        Raises:
            ValueError: If *map_* is None, or if no modelled participant is active at *frame*.
        """

        if map_ is None:
            raise ValueError("SMART needs a map to tokenize; map_ is None.")
        center_id = None if agent_ids is None else next(iter(agent_ids), None)
        batch = self.builder.build(participants, map_, frame, center_id=center_id)
        prediction = self.policy.predict_batch(batch)
        return dataclasses.replace(prediction, frames=self._future_frames(participants, batch))

    def _future_frames(
        self, participants: Dict[object, object], batch: SmartTokenBatch
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
        count = self.config.future_steps
        ahead = [frame for frame in _observed_frames(participants) if frame > anchor][:count]
        if len(ahead) == count:
            return ahead
        return ahead + [anchor + step * (index + 1) for index in range(len(ahead), count)]

    def run_closed_loop(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms0: int = 0,
        warmup_steps: Optional[int] = None,
        planning_interval: int = 10,
        scenario_steps: Optional[int] = None,
    ) -> SmartRollingResult:
        """Replay a scenario closed-loop and return its outcome.

        A thin wrapper over :class:`tactics2d.behavior.smart.rolling.SmartRollingRunner`.

        Args:
            participants (Dict[object, object]): All participants in the scenario.
            map_ (Optional[Map]): The map to tokenize.
            ego_id (object): The agent the loop is centred on.
            frame_ms0 (int, optional): Timestamp of index 0, in milliseconds. Defaults to 0.
            warmup_steps (Optional[int], optional): Steps of ground truth before the first
                replan. Defaults to None, which uses the history length.
            planning_interval (int, optional): Steps between replans. Defaults to 10.
            scenario_steps (Optional[int], optional): Number of scenario steps. Defaults to
                None, which uses the model's token span.

        Returns:
            The closed-loop outcome.
        """

        runner = SmartRollingRunner(
            self,
            config=self.config,
            warmup_steps=warmup_steps,
            planning_interval=planning_interval,
            scenario_steps=scenario_steps,
        )
        return runner.run(participants, map_, ego_id, frame_ms0=frame_ms0)
