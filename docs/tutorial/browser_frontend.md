# Browser Frontend

The browser frontend provides an interactive viewer for Tactics2D scenarios. It can
show live frames sent from Python code, preview Lanelet2/OSM maps, and replay a
short dataset clip for quick inspection.

The frontend renders the geometry supplied by the selected sensor payload. It does
not infer missing lane polygons, road surfaces, or lane topology from trajectory
metadata.

## Start the Frontend

Run the server from any environment where `tactics2d` is installed:

```bash
tactics2d start
```

The command opens the browser by default and serves the frontend at
`http://127.0.0.1:8765`. To keep the server running in the background:

```bash
tactics2d start --background
```

Use `--no-open` when running on a remote machine or in CI:

```bash
tactics2d start --background --no-open
```

Check or stop a background server with:

```bash
tactics2d status
tactics2d stop
```

The page opens on the live view by default. In live mode the frontend displays the
latest frame it receives; it is not a seekable recording player.

## Preview a Demo, Map, or Dataset

The `preview` command starts the server when needed and publishes a preview frame
or stream to the browser.

```bash
tactics2d preview demo
```

Preview an OSM or Lanelet2 map:

```bash
tactics2d preview map path/to/map.osm --lanelet2
```

Preview a supported LevelX dataset clip:

```bash
tactics2d preview dataset \
  --dataset highD \
  --folder /path/to/highD/data \
  --file 11 \
  --frames 300 \
  --follow-id 44
```

If the dataset recording has a registered map configuration, the frontend resolves
the matching OSM file automatically. Otherwise pass `--osm path/to/map.osm`.

## Send Live Frames from Python

Use `FrontendServer` when your script should own the browser server lifecycle:

```python
from tactics2d.display.renderers.web import FrontendServer


vehicle_shape = [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]

with FrontendServer(max_fps=30, open_browser=True) as renderer:
    for frame in range(120):
        sensor = {
            "id": "ego-camera",
            "perception_range": 50,
            "viewport_aspect": 16 / 9,
            "position": [frame * 0.2, 0.0],
            "yaw": 0.0,
            "frame": frame,
            "map_data": {
                "road_id_to_remove": [],
                "road_elements": [],
            },
            "participant_data": {
                "participant_id_to_remove": [],
                "participants": [
                    {
                        "id": 1,
                        "shape": "polygon",
                        "geometry": vehicle_shape,
                        "position": [frame * 0.2, 0.0],
                        "rotation": 0.0,
                        "color": "vehicle",
                        "type": "vehicle",
                        "line_width": 1,
                    }
                ],
            },
        }
        renderer.send_frame([sensor], frame=frame, layout="grid")
```

If the server is already running, connect to it directly:

```python
from tactics2d.display.renderers.web import FrontendRenderer


renderer = FrontendRenderer(host="127.0.0.1", port=8765, max_fps=30)
renderer.wait_until_ready(timeout=5.0)
renderer.send_frame([sensor], frame=frame)
```

Custom simulations usually build the `map_data` and `participant_data` dictionaries
from a `BEVCamera` update result, then pass those dictionaries through
`FrontendRenderer.send_frame`.

## Practical Notes

- Use live mode for running simulations or homework scripts that continuously push
  frames from Python.
- Use dataset preview for quick inspection of LevelX recordings.
- Use map preview to check whether a map parser produced renderable lane, area,
  junction, and roadline geometry.
- The frontend preserves the latest live snapshot when entering live mode, so a
  browser refresh does not clear the current scene while a simulation is running.
- If the browser shows only vehicles and no road, inspect the source map payload
  first. Some datasets provide lane IDs in trajectories without lane geometry that
  can be rendered as a map.

## Unified Display Backend

Tactics2D provides a unified `DisplayBackend` interface that wraps all rendering
modes (pygame, browser, matplotlib, null) behind a single API. This is the
recommended way to render scenes in new code.

### Using the Factory

```python
from tactics2d.display import (
    CameraMetadata,
    ParticipantElement,
    SceneSnapshot,
    create_display_backend,
)

# Create a browser backend (auto-starts the server if needed)
backend = create_display_backend("browser")

# Build a snapshot
snapshot = SceneSnapshot(
    frame=0,
    participants={
        1: ParticipantElement(
            id_=1, shape="polygon",
            geometry=[[-2,-1],[2,-1],[2,1],[-2,1]],
            position=(0, 0), rotation=0.0, type_="vehicle",
        ),
    },
    cameras=[CameraMetadata(id_="cam0", position=(0,0), yaw=0.0, perception_range=50)],
)

# Render on each frame
for frame in range(120):
    snapshot.frame = frame
    snapshot.participants[1].position = (frame * 0.2, 0.0)
    backend.render(snapshot)

backend.close()
```

### Integration with Environments

Set `render_mode="browser"` when creating an environment. The env automatically
creates a browser backend and sends snapshots on each `render()` call:

```python
from tactics2d.envs import ParkingEnv

env = ParkingEnv(render_mode="browser", render_fps=30)
obs, info = env.reset()

for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()              # sends frame to browser
    if terminated or truncated:
        break

env.close()
```

The browser backend manages the server lifecycle automatically:

- If a server is already running (e.g. started via `tactics2d start`), it
  connects to it.
- If no server is running, it starts one as a subprocess.
- Calling `backend.close()` or `env.close()` disconnects from the server but
  does **not** stop an externally-managed server.

Other supported render modes:

| Mode | Backend | Output |
|------|---------|--------|
| `"human"` | pygame window | Local on-screen window |
| `"rgb_array"` | pygame (off-screen) | NumPy array from `render()` |
| `"browser"` | Browser HTTP + WebSocket | Remote browser tab |
| `"matplotlib"` | Matplotlib figure | NumPy array or saved image |
| `"none"` | NullBackend (no-op) | `None` |

### Recording Output

Wrap any backend with a `GifRecorder` or `FrameExporter` to capture rendered
frames as a GIF animation or PNG sequence:

```python
from tactics2d.display import create_display_backend, SceneSnapshot
from tactics2d.display.recorder import GifRecorder

backend = create_display_backend("matplotlib")
recorder = GifRecorder(backend, output_path="output.gif", fps=10)

for frame in range(100):
    snapshot = build_snapshot(frame)
    recorder.render(snapshot)    # renders and records

recorder.save()    # writes output.gif
recorder.close()
```

Dependencies:

- **GIF export**: install `imageio` (`pip install imageio`)
- **PNG sequence export**: install `Pillow` (`pip install Pillow`)
- **Browser backend**: no extra installation needed — `fastapi`, `uvicorn`, and
  `orjson` are core dependencies of tactics2d and are installed with the package
