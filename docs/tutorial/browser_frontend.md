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
from tactics2d.frontend import FrontendServer


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
from tactics2d.frontend import FrontendRenderer


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
