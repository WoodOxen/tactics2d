# Behavior models

All integrated behavior models implement the same two entry points:

```python
trajectories = model.predict(participants, map_, frame, agent_ids=None)
result = model.rollout(
    participants,
    map_,
    ego_id,
    start_frame=None,
    controlled_ids=None,
)
```

`frame` and `start_frame` are timestamps in milliseconds. `predict` returns a
trajectory for each modeled agent. `rollout` returns a `BehaviorRolloutResult`
with the common fields `ego_id`, `frames`, `controlled_ids`, `trajectories`,
`cycles`, and `metrics`. A concrete model may return a result subclass with
additional diagnostics and expose additional keyword-only rollout options.
`metrics` is `None` when a rollout mode does not calculate closed-loop metrics;
an absent metric must not be interpreted as a zero-valued measurement.

::: tactics2d.behavior
