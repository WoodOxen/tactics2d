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

## 数据集自动发现

服务端会扫描可配置的数据根目录，查找按官方结构存放的 LevelX 数据集
（`<dataset>/data/<id>_tracks.csv`，或录像文件直接位于数据集目录下）以及可用的
OSM 地图。检测到的数据集、录像和地图会以两级下拉的形式出现在浏览器表单中，加载
场景无需手动输入任何路径。录像按其注册的地图位置分组显示，地图先选数据集再选
具体地图，对应地图自动解析。每个表单的"更多"折叠区仍保留手动路径输入作为兜底。

数据根目录按以下优先级解析：

1. `tactics2d start --data-root /path/to/datasets`
2. 环境变量 `TACTICS2D_DATA_ROOT`（多个根目录可用系统路径分隔符分隔）
3. 工作目录下的 `./data`（仓库约定）

```bash
tactics2d start --data-root /mnt/datasets
# 或
export TACTICS2D_DATA_ROOT=/mnt/datasets
tactics2d start
```

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

## 录像

提供三种互补的录像方式：

### 浏览器录屏

底部栏的"录屏"按钮通过浏览器 MediaRecorder API 把渲染画面录制为 `.webm` 视频，
停止时自动下载。它录制的是你实际看到的内容——当前布局下所有传感器小窗的合成
画面——实时推流、demo 和数据集预览都可以录。录制帧率受浏览器渲染速度限制。

### 帧数据录制与回放（服务端）

实时面板的"帧录制"控件把服务端发布的每一帧 payload 逐行写入 JSONL 文件。录制
开始时会先写入当前场景快照，保证回放能还原完整场景；录制不受浏览器丢帧影响。
保存的录制会出现在"回放录制"下拉框中，通过与数据集预览相同的管线回放（暂停、
停止和进度条的行为一致）。

录制默认保存在 `~/.cache/tactics2d/recordings/`，可用 `TACTICS2D_RECORD_DIR`
环境变量覆盖。HTTP 接口：

| 接口 | 用途 |
|---|---|
| `POST /api/record/start` | 开始录制（可选 `{"name": ...}`） |
| `POST /api/record/stop` | 停止并保存录制 |
| `GET /api/recordings` | 列出已保存的录制 |
| `POST /api/preview/replay` | 回放：`{"name": ..., "max_fps": 30, "loop": false}` |

### Python 端离线渲染

需要像素级精确的 GIF 或 PNG 输出时，把同一个 `SceneSnapshot` 并行渲染到包装了
录制器的 matplotlib 后端：

```python
from tactics2d.display import create_display_backend
from tactics2d.display.recorder import GifRecorder

browser = create_display_backend("browser")
recorder = GifRecorder(create_display_backend("matplotlib"), "output.gif", fps=10)

for frame in range(300):
    snapshot = build_snapshot(frame)
    browser.render(snapshot)   # 浏览器实时查看
    recorder.render(snapshot)  # 离线渲染 GIF

recorder.save()
recorder.close()
browser.close()
```

## 使用提示

- 实时模式适合正在运行的仿真脚本或课程作业脚本，Python 端持续向浏览器推帧。
- 数据集预览适合快速检查 LevelX 录像。
- 地图预览适合确认地图解析器是否生成了可渲染的 lane、area、junction 和 roadline。
- 进入实时模式时，前端会保留最近一次实时快照；仿真运行中刷新浏览器不会清空当前场景。
- 如果浏览器只显示车辆、没有道路，先检查源地图 payload。部分数据集在轨迹中提供
  `Lane_ID`，但并没有可渲染为地图的车道几何。
