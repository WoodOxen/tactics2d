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

### 远程模式

服务器默认绑定 `127.0.0.1`，只接受本机连接。仿真跑在实验室服务器上、想在自己
工位的浏览器里查看时，绑定所有网卡并通过服务器地址访问：

```bash
# 仿真服务器上
tactics2d start --host 0.0.0.0 --port 8765 --background --no-open

# 工位上：浏览器打开 http://<服务器IP>:8765
```

Python 发布端同样按地址连接：

```python
renderer = FrontendRenderer(host="<服务器IP>", port=8765)
```

所有 `preview` 子命令也接受 `--host`/`--port`，因此
`tactics2d preview dataset --host 0.0.0.0 ...` 可以直接在远程机器上使用。

> **安全提示**：服务器没有任何鉴权——能访问端口的人既能看画面也能推送自己的
> 帧。只在可信内网中暴露端口；或者保持绑定 `127.0.0.1`，通过 SSH 隧道访问：
> `ssh -L 8765:127.0.0.1:8765 user@server`，然后在本地浏览
> `http://127.0.0.1:8765`。不要把端口转发到公网。

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

预览 NuPlan 日志（城市地图会根据日志自动解析，UTM 量级的坐标在渲染前会平移到局部
原点）：

```bash
tactics2d preview dataset \
  --dataset NuPlan \
  --folder /path/to/nuPlan/data/cache \
  --file mini/2021.10.05.07.10.04_veh-52_01442_01802.db \
  --frames 300
```

## 数据集自动发现

服务端会扫描可配置的数据根目录，查找预览管线支持的所有数据集家族以及可用的
OSM 地图。检测到的数据集、录像和地图会以两级下拉的形式出现在浏览器表单中，
加载场景无需手动输入任何路径。每个表单的"更多"折叠区仍保留手动路径输入作为兜底。

| 数据集 | 数据根目录下的期望结构 | 地图 |
|---|---|---|
| highD/inD/rounD/exiD/uniD | `<dataset>/data/<id>_tracks.csv`（官方结构） | 注册的 OSM 配置，自动解析 |
| NuPlan | `nuPlan/data/cache/<split>/*.db`（或 nuplan-devkit 结构） | 同级 `maps/` 中的 geopackage 城市地图，按日志自动匹配 |
| NGSIM | `NGSIM/<location>/trajectories*.csv` | csv 同级 `gis-files/*.shp` 街道中心线（航向由运动方向推导） |
| INTERACTION | `INTERACTION*/recorded_trackfiles/<scenario>/vehicle_tracks_XXX.csv` | `recorded_trackfiles/` 同级的 `maps/<scenario>.osm` |
| DLP | `DLP/data/DJI_XXXX_*.json` | 数据目录内或上级的 `DLP.osm` |
| CitySim | `CitySim/**/*.csv` | 无 |
| Argoverse 2 | `Argoverse2/<split>/<scenario>/`（parquet + `log_map_archive_*.json`） | 场景目录内的地图 json |
| DriveInsightD | `DriveInsightD/<id>_scenario.xosc` | 同目录任意 `.xodr`（可选） |
| WOMD | `WOMD/**/*.tfrecord*` 分片 | 分片内自带的逐场景矢量地图 |

LevelX 录像按其注册的地图位置分组显示；路径式录像（NuPlan、NGSIM、INTERACTION、
CitySim、Argoverse 2、WOMD）按顶层目录分组。不超过 32 MB 的 WOMD 分片会按场景
枚举为 `<分片>/<场景id>` 条目（枚举场景 id 需要解码整个文件）；官方尺寸的大分片
只显示一个条目、默认预览第一个场景，可在 CLI 用
`--file "<分片>/<场景id>"` 指定其它场景。

注意事项：没有规则帧网格的数据集（NuPlan 约 20 Hz 的激光扫描、DriveInsightD 的
场景顶点）按实际观测到的时间戳推进；未指定 `--follow-id` 时相机跟随存活最久的
车辆；UTM/州平面量级的全局坐标（NuPlan、NGSIM）渲染前会平移到局部原点以保证
float32 精度；锥桶、护栏等静态目标会渲染为按类型着色的小标记（数据集提供尺寸时
为小矩形，否则为圆点）。地图表单中单独预览
`.gpkg` 城市地图时只显示地图中心附近的一个窗口，以保证浏览器端载荷可控。

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

### 多环境共用一个服务器

多个环境（包括不同进程）可以同时向同一个服务器推流，各自渲染到独立视图。
每个发布端使用唯一的传感器 id，并传 `remove_missing_sensors=False`，避免一个
环境的帧把另一个环境的视图从浏览器删除：

```python
renderer.send_frame([{"id": "env-a", ...}], frame=frame, remove_missing_sensors=False)
```

使用 `BrowserBackend` 时通过构造参数表达同样的语义：
`BrowserBackend(sensor_id="env-a", exclusive=False)`；非独占的 backend 在
`close()` 时也只移除自己的视图。帧确认在服务端按发布序号跟踪，因此各自独立
编号帧的并发流不会干扰彼此的 `wait_ack`/`drop_if_busy` 背压。

## 录像

提供三种互补的录像方式：

### 浏览器录屏

"录屏"按钮通过浏览器 MediaRecorder API 录制渲染画面。勾选"兼容模式"（默认开启）时，
停止后录制会上传到服务端，由 ffmpeg 整理为恒定帧率、尺寸按 4 像素对齐的 H.264 MP4
再下载——MediaRecorder 的原始输出是可变帧率（声明为 0/1），GNOME Videos 等严格的
播放器会拒绝播放；非对齐的画面宽度还会让部分硬件解码器（如 GStreamer VA-API）崩溃。
采集本身优先使用 VP9 编码——浏览器的实时 H.264 编码器会在运动车辆后面留下残块拖影，
VP9 没有这个问题。取消勾选则跳过转码直接下载原始录制（更快；多数浏览器为 WebM，
Safari 为 MP4）。
ffmpeg 无需额外安装：从 `PATH` 或核心依赖 `imageio-ffmpeg` 自带的二进制解析，两者都
缺失时自动回退为原始录制。它录制
的是你实际看到的内容——当前布局下所有传感器小窗的合成画面——实时推流、demo 和
数据集预览都可以录。录制帧率受浏览器渲染速度限制。

### 帧数据录制与回放（服务端）

侧栏"录制"区的"帧录制"控件把服务端发布的每一帧 payload 逐行写入 JSONL 文件。录制
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

### 100 Hz 帧率

`FrontendRenderer` 经实测可以稳定以 100 Hz（`max_fps` 上限）推流：keep-alive
连接下，本机回环上一帧 100 个参与者 + 400 个道路元素的 HTTP 往返不到 1 毫秒。
高频循环实际能跑多快取决于两个设置：

- **`wait_ack=True`（默认）**：每帧等待浏览器的渲染确认，因此稳定在浏览器的
  实际刷新率（普通显示器 60 fps）——这是有意的背压设计，仿真不会跑在屏幕
  前面。
- **`wait_ack=False`**：按设定帧率全速发布；浏览器按自身刷新率渲染，多余的
  帧按流合并（计入"跳帧"角标），100 Hz 的控制循环不会被显示端拖慢。

回环实测：`wait_ack=False` 时持续 99.6 fps（500/500 帧全部送达，最大抖动约
1 ms），同时浏览器稳定渲染 60 fps。
