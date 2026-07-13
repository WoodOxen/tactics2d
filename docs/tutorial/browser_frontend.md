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

### Remote mode

The server binds to `127.0.0.1` by default, which only accepts local
connections. To view the frontend from another machine — a common setup when
simulations run on a lab server — bind to all interfaces and open the page via
the server's address:

```bash
# On the simulation server
tactics2d start --host 0.0.0.0 --port 8765 --background --no-open

# On your workstation: browse to http://<server-ip>:8765
```

Python publishers reach the same server by address:

```python
renderer = FrontendRenderer(host="<server-ip>", port=8765)
```

Every `preview` subcommand also accepts `--host`/`--port`, so
`tactics2d preview dataset --host 0.0.0.0 ...` works on a remote box.

> **Security note**: the server has no authentication — anyone who can reach
> the port can watch frames and publish their own. Only expose it on a trusted
> intranet, or keep it on `127.0.0.1` and reach it through an SSH tunnel:
> `ssh -L 8765:127.0.0.1:8765 user@server`, then browse
> `http://127.0.0.1:8765` locally. Do not forward the port on a public
> interface.

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

Preview a NuPlan log (the city map is resolved from the log automatically and the
UTM-scale coordinates are shifted to a local origin before rendering):

```bash
tactics2d preview dataset \
  --dataset NuPlan \
  --folder /path/to/nuPlan/data/cache \
  --file mini/2021.10.05.07.10.04_veh-52_01442_01802.db \
  --frames 300
```

## Dataset Discovery

The server scans a configurable data root for every dataset family the preview
pipeline supports and for available OSM maps. Detected datasets, recordings,
and maps appear as dropdowns in the browser forms, so scenes can be loaded
without typing any path. Manual path fields remain available under the advanced
("更多") section of each form as a fallback.

| Dataset | Expected layout under the data root | Map |
|---|---|---|
| highD/inD/rounD/exiD/uniD | `<dataset>/data/<id>_tracks.csv` (official) | registered OSM configs, resolved automatically |
| NuPlan | `nuPlan/data/cache/<split>/*.db` (or the nuplan-devkit layout) | geopackage city maps in a sibling `maps/`, auto-resolved from the log |
| NGSIM | `NGSIM/<location>/trajectories*.csv` | `gis-files/*.shp` street centerlines next to the csv (headings derived from motion) |
| INTERACTION | `INTERACTION*/recorded_trackfiles/<scenario>/vehicle_tracks_XXX.csv` | `maps/<scenario>.osm` next to `recorded_trackfiles/` |
| DLP | `DLP/data/DJI_XXXX_*.json` | `DLP.osm` in or next to the data folder |
| CitySim | `CitySim/**/*.csv` | none |
| Argoverse 2 | `Argoverse2/<split>/<scenario>/` (parquet + `log_map_archive_*.json`) | per-scenario map json |
| DriveInsightD | `DriveInsightD/<id>_scenario.xosc` | any `.xodr` in the same folder (optional) |
| WOMD | `WOMD/**/*.tfrecord*` shards | per-scenario vector map inside the shard |

LevelX recordings are grouped by their registered map location; path-style
recordings (NuPlan, NGSIM, INTERACTION, CitySim, Argoverse 2, WOMD) are grouped
by their top-level folder. WOMD shards up to 32 MB are listed per scenario as
`<shard>/<scenario_id>` (enumerating ids means decoding the whole file);
official-size shards appear as one entry and preview their first scenario —
pass `--file "<shard>/<scenario_id>"` on the CLI to pick another.

Notes: datasets without a regular frame grid (NuPlan's ~20 Hz lidar sweeps,
DriveInsightD's scenario vertices) are previewed on their observed timestamps;
the camera follows the longest-lived vehicle unless `--follow-id` is given;
global UTM/state-plane coordinates (NuPlan, NGSIM) are shifted to a local
origin before rendering to stay within float32 resolution; cones, barriers,
and other static objects render as small color-coded markers (rectangles when
the dataset provides dimensions, dots otherwise). Standalone NuPlan map
previews (`.gpkg` in the map form) show a capped window around the city center
to keep the payload browser-friendly.

The data root is resolved in this order:

1. `tactics2d start --data-root /path/to/datasets`
2. The `TACTICS2D_DATA_ROOT` environment variable (multiple roots can be
   separated by the OS path separator)
3. `./data` relative to the working directory (repository convention)

```bash
tactics2d start --data-root /mnt/datasets
# or
export TACTICS2D_DATA_ROOT=/mnt/datasets
tactics2d start
```

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

### Multiple environments on one server

Several environments (separate processes included) can stream to the same
server at once, each rendering into its own view. Give each publisher a unique
sensor id and pass `remove_missing_sensors=False` so one environment's frames
do not remove the other's view:

```python
# Environment A                          # Environment B
renderer.send_frame(                     renderer.send_frame(
    [{"id": "env-a", ...}],                  [{"id": "env-b", ...}],
    frame=frame,                             frame=frame,
    remove_missing_sensors=False,            remove_missing_sensors=False,
)                                        )
```

With `BrowserBackend`, the same is expressed through the constructor:
`BrowserBackend(sensor_id="env-a", exclusive=False)`. A non-exclusive backend
also removes only its own view on `close()`. Frame acknowledgements are
tracked per published frame server-side, so concurrent streams that number
their frames independently do not interfere with each other's
`wait_ack`/`drop_if_busy` backpressure.

## Recording

Three complementary recording options are available:

### Screen recording (browser)

The 录屏 button records the rendered sensor windows via the browser's
MediaRecorder API. With compatibility mode enabled (the default), the capture
is finalized server-side with ffmpeg into a constant-frame-rate H.264 MP4
with 4-pixel-aligned dimensions — raw MediaRecorder files declare a variable
frame rate (0/1) that strict players such as GNOME Videos refuse to play,
and misaligned frame widths crash some hardware-accelerated decoders
(e.g. GStreamer VA-API). The capture itself
prefers VP9, whose realtime encoder — unlike the browser's H.264 one —
leaves no ghost trails behind moving vehicles. Untick 兼容模式 to skip the
transcode and download the raw recording directly (faster; WebM on most
browsers, MP4 on Safari). ffmpeg needs no extra installation: it is
resolved from `PATH` or from the bundled `imageio-ffmpeg` core dependency, and
the raw file is used automatically when neither is available.
It captures exactly what you see — all sensor tiles composited in the current
layout — and works for live streams, demos, and dataset previews alike. The
capture frame rate is bounded by the browser's rendering speed.

### Frame recording and replay (server)

The 帧录制 controls in the sidebar 录制 section capture every frame payload
published by the server into a JSONL file, one line per frame. Recording starts with the
current scene snapshot so a replay reproduces the full scene, and it is not
thinned when the browser drops frames. Saved recordings appear in the 回放
dropdown and stream back through the same pipeline as dataset previews
(pause, stop, and the progress bar work the same way).

Recordings are stored in `~/.cache/tactics2d/recordings/` by default; set
`TACTICS2D_RECORD_DIR` to override. The HTTP API:

| Endpoint | Purpose |
|---|---|
| `POST /api/record/start` | Start recording (optional `{"name": ...}`) |
| `POST /api/record/stop` | Stop and save the recording |
| `GET /api/recordings` | List saved recordings |
| `POST /api/preview/replay` | Replay: `{"name": ..., "max_fps": 30, "loop": false}` |

### Offline rendering (Python)

For pixel-perfect GIF or PNG output, render the same `SceneSnapshot` to a
matplotlib backend wrapped in a recorder, in parallel with the browser view:

```python
from tactics2d.display import create_display_backend
from tactics2d.display.recorder import GifRecorder

browser = create_display_backend("browser")
recorder = GifRecorder(create_display_backend("matplotlib"), "output.gif", fps=10)

for frame in range(300):
    snapshot = build_snapshot(frame)
    browser.render(snapshot)   # live view in the browser
    recorder.render(snapshot)  # offline GIF rendering

recorder.save()
recorder.close()
browser.close()
```

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

### Frame rates up to 100 Hz

`FrontendRenderer` sustains a measured 100 Hz publish rate (the `max_fps`
ceiling) over its keep-alive connection: on loopback, one frame with 100
participants and 400 road elements costs well under a millisecond of HTTP
round-trip. Two settings decide what a high-rate loop actually achieves:

- **`wait_ack=True` (default)** throttles each frame to the browser's real
  render pace. Browsers repaint on `requestAnimationFrame`, so this settles at
  the display refresh rate (60 fps on a typical monitor) — by design, the
  simulation then never outruns what the screen can show.
- **`wait_ack=False`** publishes at the full requested rate; the browser
  renders at its refresh rate and coalesces the surplus per stream (counted in
  the 跳帧 badge), so a 100 Hz control loop keeps its timing without the
  display holding it back.

Measured on loopback: 99.6 fps sustained publishing with `wait_ack=False`
(500/500 frames delivered, max jitter ≈ 1 ms) while the browser rendered a
steady 60 fps.

## Unified Display Backend

Tactics2D provides a unified `DisplayBackend` interface that wraps all rendering
modes (pygame, browser, matplotlib, null) behind a single API. This is the
recommended way to render scenes in new code.

### Using the Factory

```python
from tactics2d.display import create_display_backend, SceneSnapshot

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
- **Browser backend**: install `fastapi` and `uvicorn` (included as optional
  dependencies of tactics2d)
