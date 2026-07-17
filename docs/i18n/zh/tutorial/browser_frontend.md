# 浏览器前端

浏览器前端提供了一个用于查看 Tactics2D 场景的交互式页面。它可以显示 Python
代码实时发送的帧，也可以预览 Lanelet2/OSM 地图，或快速回放一小段数据集片段。

前端只渲染传入的传感器几何数据。它不会自动推断缺失的车道
多边形、道路面层或车道拓扑。

## 启动前端

在已经安装 `tactics2d` 的环境中运行：

```bash
tactics2d start
```

该命令默认打开浏览器，并在 `http://127.0.0.1:8765` 提供前端页面。需要让服务在后台
运行时使用：

```bash
tactics2d start --background
```

在远程服务器或 CI 环境中，可以用 `--no-open` 禁止自动打开浏览器：

```bash
tactics2d start --background --no-open
```

查看或停止后台服务：

```bash
tactics2d status
tactics2d stop
```

页面默认进入实时视图。实时模式会显示最新收到的一帧，它不是可拖动进度条的录像播放器。

## 预览示例、地图或数据集

`preview` 命令会在需要时自动启动前端服务，并向浏览器发布预览帧或预览流。

```bash
tactics2d preview demo
```

预览 OSM 或 Lanelet2 地图：

```bash
tactics2d preview map path/to/map.osm --lanelet2
```

预览支持的 LevelX 数据集片段：

```bash
tactics2d preview dataset \
  --dataset highD \
  --folder /path/to/highD/data \
  --file 11 \
  --frames 300 \
  --follow-id 44
```

如果该数据集录像有已注册的地图配置，前端会自动解析对应的 OSM 文件。否则需要显式传入
`--osm path/to/map.osm`。

## 从 Python 实时发送帧

如果脚本希望自己管理浏览器服务的生命周期，可以使用 `FrontendServer`：

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

如果前端服务已经启动，可以直接连接：

```python
from tactics2d.display.renderers.web import FrontendRenderer


renderer = FrontendRenderer(host="127.0.0.1", port=8765, max_fps=30)
renderer.wait_until_ready(timeout=5.0)
renderer.send_frame([sensor], frame=frame)
```

自定义仿真通常会先通过 `BEVCamera.update` 得到 `map_data` 和 `participant_data`，再把
这些字典传给 `FrontendRenderer.send_frame`。

## 使用提示

- 实时模式适合正在运行的仿真脚本或课程作业脚本，Python 端持续向浏览器推帧。
- 数据集预览适合快速检查 LevelX 录像。
- 地图预览适合确认地图解析器是否生成了可渲染的 lane、area、junction 和 roadline。
- 进入实时模式时，前端会保留最近一次实时快照；仿真运行中刷新浏览器不会清空当前场景。
- 如果浏览器只显示车辆、没有道路，先检查源地图 payload。部分数据集在轨迹中提供
  `Lane_ID`，但并没有可渲染为地图的车道几何。

## 统一显示后端

Tactics2D 提供统一的 `DisplayBackend` 接口，把所有渲染方式（pygame、浏览器、
matplotlib、null）封装在同一套 API 后面。新代码推荐使用这种方式渲染场景。

### 使用工厂函数

```python
from tactics2d.display import (
    CameraMetadata,
    ParticipantElement,
    SceneSnapshot,
    create_display_backend,
)

# 创建浏览器后端（需要时自动启动前端服务）
backend = create_display_backend("browser")

# 构造一帧快照
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

# 每帧渲染
for frame in range(120):
    snapshot.frame = frame
    snapshot.participants[1].position = (frame * 0.2, 0.0)
    backend.render(snapshot)

backend.close()
```

### 与环境集成

创建环境时设置 `render_mode="browser"`，环境会自动创建浏览器后端，并在每次
`render()` 调用时发送快照：

```python
from tactics2d.envs import ParkingEnv

env = ParkingEnv(render_mode="browser", render_fps=30)
obs, info = env.reset()

for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()              # 把当前帧发送到浏览器
    if terminated or truncated:
        break

env.close()
```

浏览器后端会自动管理服务生命周期：

- 如果服务已经在运行（例如通过 `tactics2d start` 启动），后端会直接连接它。
- 如果没有服务在运行，后端会以子进程方式启动一个。
- 调用 `backend.close()` 或 `env.close()` 只会断开连接，**不会**停止由外部管理的服务。

其他支持的渲染模式：

| 模式 | 后端 | 输出 |
|------|------|------|
| `"human"` | pygame 窗口 | 本地屏幕窗口 |
| `"rgb_array"` | pygame（离屏） | `render()` 返回 NumPy 数组 |
| `"browser"` | 浏览器 HTTP + WebSocket | 远程浏览器标签页 |
| `"matplotlib"` | Matplotlib 图像 | NumPy 数组或保存的图片 |
| `"none"` | NullBackend（空操作） | `None` |

### 录制输出

用 `GifRecorder` 或 `FrameExporter` 包装任意后端，可以把渲染帧保存为 GIF 动画
或 PNG 序列：

```python
from tactics2d.display import create_display_backend, SceneSnapshot
from tactics2d.display.recorder import GifRecorder

backend = create_display_backend("matplotlib")
recorder = GifRecorder(backend, output_path="output.gif", fps=10)

for frame in range(100):
    snapshot = build_snapshot(frame)
    recorder.render(snapshot)    # 渲染并记录

recorder.save()    # 写出 output.gif
recorder.close()
```

依赖说明：

- **GIF 导出**：安装 `imageio`（`pip install imageio`）
- **PNG 序列导出**：安装 `Pillow`（`pip install Pillow`）
- **浏览器后端**：无需额外安装——`fastapi`、`uvicorn` 和 `orjson` 是 tactics2d
  的核心依赖，随包一起安装
