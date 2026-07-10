import * as THREE from "three";

const DEFAULT_VIEWPORT_ASPECT = 16 / 9;

const COLORS = {
  area: "#2f3542",
  black: "#101418",
  bicycle: "#fd9644",
  bus: "#4b6584",
  cyclist: "#fd9644",
  default: "#a5b1c2",
  heading_arrow: "#101418",
  hole: "#f4f7f7",
  highway: "#2f3542",
  lane: "#2f3542",
  "light-blue": "#45aaf2",
  "light-turquoise": "#2bcbba",
  pedestrian: "#45aaf2",
  road: "#2f3542",
  roadline: "#f4f7f7",
  vehicle: "#2bcbba",
  white: "#f4f7f7"
};

const ORDERS = {
  road: 1,
  lane: 2,
  hole: 3,
  roadline: 4,
  solid: 4,
  dashed: 4,
  vehicle: 6,
  cyclist: 6,
  pedestrian: 6,
  heading_arrow: 7
};

function colorFor(value, typeName) {
  if (value && value.startsWith && value.startsWith("#")) return value;
  return COLORS[value] || COLORS[typeName] || COLORS.default;
}

function orderFor(typeName) {
  if (typeof typeName === "number") return typeName;
  return ORDERS[typeName] || 1;
}

function makeShape(points) {
  const shape = new THREE.Shape();
  points.forEach((point, index) => {
    if (index === 0) shape.moveTo(point[0], point[1]);
    else shape.lineTo(point[0], point[1]);
  });
  return shape;
}

function parseOptionalNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  return Number(value);
}

function parseOptionalText(value) {
  if (value === null || value === undefined) return null;
  const trimmed = value.trim();
  return trimmed.length ? trimmed : null;
}

class SensorView {
  constructor(sensor) {
    this.id = sensor.id;
    this.perceptionRange = sensor.perception_range || 50;
    this.viewportAspect = sensor.viewport_aspect || DEFAULT_VIEWPORT_ASPECT;
    this.roadObjects = new Map();
    this.participantObjects = new Map();

    this.element = document.createElement("section");
    this.element.className = "sensor-view";
    this.element.id = sensor.id;

    this.label = document.createElement("div");
    this.label.className = "sensor-label";
    this.label.textContent = sensor.id;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.hole);
    // preserveDrawingBuffer keeps the frame readable for screen recording drawImage().
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.addEventListener("webglcontextrestored", () => this.resize());

    this.viewport = document.createElement("div");
    this.viewport.className = "sensor-viewport";
    this.viewport.appendChild(this.renderer.domElement);
    this.element.appendChild(this.viewport);
    this.element.appendChild(this.label);

    this.camera = new THREE.OrthographicCamera(-50, 50, 50, -50, 0.1, 1000);
    this.camera.position.set(0, 0, 200);
    this.camera.lookAt(0, 0, 0);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.element);
    this.updateView(sensor);
  }

  resize() {
    const containerWidth = Math.max(1, this.element.clientWidth);
    const containerHeight = Math.max(1, this.element.clientHeight);
    const containerAspect = containerWidth / containerHeight;
    let width = containerWidth;
    let height = containerHeight;

    if (containerAspect > this.viewportAspect) {
      width = Math.round(containerHeight * this.viewportAspect);
    } else {
      height = Math.round(containerWidth / this.viewportAspect);
    }

    this.viewport.style.width = `${width}px`;
    this.viewport.style.height = `${height}px`;
    this.renderer.setSize(width, height, false);
    this.updateCameraBounds();
    this.render();
  }

  disposeObject(mesh) {
    if (!mesh) return;
    if (mesh.geometry) mesh.geometry.dispose();
    if (mesh.material) mesh.material.dispose();
  }

  dispose() {
    this.resizeObserver.disconnect();
    this.roadObjects.forEach((mesh) => {
      this.scene.remove(mesh);
      this.disposeObject(mesh);
    });
    this.participantObjects.forEach((mesh) => {
      this.scene.remove(mesh);
      this.disposeObject(mesh);
    });
    this.renderer.dispose();
    this.element.remove();
  }

  updateCameraBounds() {
    const range = this.perceptionRange;
    this.camera.left = -range * this.viewportAspect;
    this.camera.right = range * this.viewportAspect;
    this.camera.top = range;
    this.camera.bottom = -range;
    this.camera.far = Math.max(1000, range * 8);
    this.camera.updateProjectionMatrix();
  }

  updateView(sensor) {
    const position = sensor.position || [0, 0];
    this.perceptionRange = sensor.perception_range || this.perceptionRange;
    this.viewportAspect = sensor.viewport_aspect || this.viewportAspect;
    this.camera.position.set(position[0], position[1], this.perceptionRange * 4);
    this.camera.rotation.set(0, 0, sensor.yaw || 0);
    this.updateCameraBounds();
  }

  createPolygon(element) {
    const geometry = new THREE.ShapeGeometry(makeShape(element.geometry || []));
    const material = new THREE.MeshBasicMaterial({
      color: colorFor(element.color, element.type),
      side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = orderFor(element.type);
    return mesh;
  }

  createCircle(element) {
    const geometry = new THREE.CircleGeometry(element.radius || 1, 32);
    const material = new THREE.MeshBasicMaterial({ color: colorFor(element.color, element.type) });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = orderFor(element.type);
    return mesh;
  }

  createLine(element) {
    const points = (element.geometry || []).map(([x, y]) => new THREE.Vector3(x, y, 0));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: colorFor(element.color, element.type) });
    const mesh = new THREE.Line(geometry, material);
    mesh.renderOrder = orderFor(element.type);
    return mesh;
  }

  replaceObject(targetMap, id, mesh) {
    const existing = targetMap.get(id);
    if (existing) {
      this.scene.remove(existing);
      this.disposeObject(existing);
    }
    targetMap.set(id, mesh);
    this.scene.add(mesh);
  }

  updateRoadElements(mapData) {
    (mapData.road_id_to_remove || []).forEach((id) => {
      const existing = this.roadObjects.get(id);
      if (!existing) return;
      this.scene.remove(existing);
      this.roadObjects.delete(id);
    });

    (mapData.road_elements || []).forEach((element) => {
      let mesh = null;
      if (element.shape === "polygon") mesh = this.createPolygon(element);
      if (element.shape === "line") mesh = this.createLine(element);
      if (mesh) this.replaceObject(this.roadObjects, element.id, mesh);
    });
  }

  updateParticipants(participantData) {
    (participantData.participant_id_to_remove || []).forEach((id) => {
      const existing = this.participantObjects.get(id);
      if (!existing) return;
      this.scene.remove(existing);
      this.participantObjects.delete(id);
    });

    (participantData.participants || []).forEach((participant) => {
      let mesh = this.participantObjects.get(participant.id);
      if (!mesh) {
        if (participant.shape === "polygon") mesh = this.createPolygon(participant);
        if (participant.shape === "circle") mesh = this.createCircle(participant);
        if (!mesh) return;
        this.replaceObject(this.participantObjects, participant.id, mesh);
      }

      const position = participant.position || [0, 0];
      mesh.position.set(position[0], position[1], 0);
      mesh.rotation.set(0, 0, participant.rotation || 0);
    });
  }

  update(sensor) {
    this.label.textContent = `${sensor.id} ${sensor.frame ?? ""}`;
    if (sensor.map_data) this.updateRoadElements(sensor.map_data);
    if (sensor.participant_data) this.updateParticipants(sensor.participant_data);
    this.updateView(sensor);
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}

class PreviewControls {
  constructor() {
    this.source = "live";
    this.sourceButtons = Array.from(document.querySelectorAll("[data-source-tab]"));
    this.datasetForm = document.getElementById("dataset-form");
    this.mapForm = document.getElementById("map-form");
    this.livePanel = document.getElementById("live-panel");
    this.datasetSelect = document.getElementById("dataset-select");
    this.datasetMapConfig = document.getElementById("dataset-map-config");
    this.mapConfig = document.getElementById("map-config");
    this.mapSelect = document.getElementById("map-select");
    this.mapDataset = document.getElementById("map-dataset");
    this.status = document.getElementById("preview-status");
    this.progress = document.getElementById("preview-progress");
    this.progressTrack = document.getElementById("progress-track");
    this.streamMode = document.getElementById("stream-mode");
    this.pauseButton = document.getElementById("pause-preview");
    this.recordButton = document.getElementById("record-frames");
    this.replaySelect = document.getElementById("replay-select");
    this.isPaused = false;
    this.isRecordingFrames = false;
    this.options = { levelx_datasets: [], map_configs: [], datasets: [], maps: [], defaults: {} };

    this.bind();
    this.loadOptions();
    window.setInterval(() => this.refreshStatus(), 500);
  }

  bind() {
    this.sourceButtons.forEach((button) => {
      button.addEventListener("click", () => this.showSource(button.dataset.sourceTab));
    });
    this.datasetSelect.addEventListener("change", () => this.updateDatasetFiles());
    document
      .getElementById("dataset-file")
      .addEventListener("change", () => this.updateDatasetMapConfigs());
    this.mapDataset.addEventListener("change", () => this.populateMapSelect());
    this.mapSelect.addEventListener("change", () => this.applyMapSelection());
    this.datasetForm.addEventListener("submit", (event) => {
      event.preventDefault();
      this.startDatasetPreview();
    });
    this.mapForm.addEventListener("submit", (event) => {
      event.preventDefault();
      this.loadMapPreview();
    });
    this.pauseButton.addEventListener("click", () => this.togglePause());
    document.getElementById("stop-preview").addEventListener("click", () => this.stopPreview());
    this.recordButton.addEventListener("click", () => this.toggleFrameRecording());
    document.getElementById("replay-start").addEventListener("click", () => this.startReplay());
  }

  async request(path, payload = null) {
    const options = payload
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }
      : {};
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.message || response.statusText);
    }
    return data;
  }

  async loadOptions() {
    try {
      this.options = await this.request("/api/preview/options");
      this.populateDatasetSelect();
      this.populateDefaults();
      this.updateDatasetFiles();
      this.updateMapConfigSelect();
      this.populateMapGroups();
      this.loadRecordings();
      this.showSource(this.source);
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  catalogEntry(dataset) {
    return (this.options.datasets || []).find((entry) => entry.dataset === dataset) || null;
  }

  populateDatasetSelect() {
    // Detected datasets first; fall back to the full list when nothing was found.
    const detected = (this.options.datasets || []).map((entry) => entry.dataset);
    const datasets = detected.length ? detected : this.options.levelx_datasets;
    this.datasetSelect.replaceChildren();
    datasets.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset;
      option.textContent = dataset;
      this.datasetSelect.appendChild(option);
    });
  }

  populateDefaults() {
    const defaults = this.options.defaults || {};
    if (defaults.dataset) this.datasetSelect.value = defaults.dataset;
    document.getElementById("dataset-frames").value = defaults.frames || 300;
    document.getElementById("dataset-max-fps").value = defaults.max_fps || 30;
    document.getElementById("dataset-range").value = defaults.perception_range || 80;
  }

  updateDatasetFiles() {
    const dataset = this.datasetSelect.value;
    const entry = this.catalogEntry(dataset);
    const fileSelect = document.getElementById("dataset-file");

    // Detected recordings win; otherwise offer the ids registered in map configs.
    const configs = this.options.map_configs.filter((config) => config.dataset === dataset);
    let files = entry ? entry.files : [];
    if (!files.length) {
      files = [...new Set(configs.flatMap((config) => config.trajectory_files || []))].sort(
        (a, b) => a - b
      );
    }

    // Group recordings by their registered map location.
    fileSelect.replaceChildren();
    const remaining = new Set(files);
    configs.forEach((config) => {
      const members = files.filter((fileId) =>
        (config.trajectory_files || []).includes(fileId)
      );
      if (!members.length) return;
      const group = document.createElement("optgroup");
      group.label = config.description || config.name;
      members.forEach((fileId) => {
        group.appendChild(this.fileOption(fileId));
        remaining.delete(fileId);
      });
      fileSelect.appendChild(group);
    });
    if (remaining.size) {
      const group = document.createElement("optgroup");
      group.label = "未注册位置";
      [...remaining].sort((a, b) => a - b).forEach((fileId) => {
        group.appendChild(this.fileOption(fileId));
      });
      fileSelect.appendChild(group);
    }

    document.getElementById("dataset-folder").value = entry ? entry.folder : "";
    this.updateDatasetMapConfigs();
  }

  fileOption(fileId) {
    const option = document.createElement("option");
    option.value = String(fileId);
    option.textContent = String(fileId);
    return option;
  }

  populateMapGroups() {
    const maps = this.options.maps || [];
    const groups = [...new Set(maps.map((map) => map.dataset || "其他"))];
    this.mapDataset.replaceChildren();
    groups.forEach((group) => {
      const option = document.createElement("option");
      option.value = group;
      option.textContent = group;
      this.mapDataset.appendChild(option);
    });
    this.populateMapSelect();
  }

  populateMapSelect() {
    const group = this.mapDataset.value;
    const maps = (this.options.maps || []).filter(
      (map) => (map.dataset || "其他") === group
    );
    this.mapSelect.replaceChildren(this.blankOption("手动输入路径"));
    maps.forEach((map) => {
      const option = document.createElement("option");
      option.value = map.osm_path;
      option.textContent = map.dataset
        ? `${map.name} ${map.description || ""}`.trim()
        : map.name;
      if (map.dataset) option.dataset.configName = map.name;
      this.mapSelect.appendChild(option);
    });
    if (maps.length) {
      this.mapSelect.value = maps[0].osm_path;
      this.applyMapSelection();
    }
  }

  applyMapSelection() {
    const selected = this.mapSelect.selectedOptions[0];
    if (!selected || !selected.value) return;
    document.getElementById("map-osm").value = selected.value;
    this.mapConfig.value = selected.dataset.configName || "";
  }

  updateDatasetMapConfigs() {
    const dataset = this.datasetSelect.value;
    const configs = this.options.map_configs.filter((config) => config.dataset === dataset);
    this.datasetMapConfig.replaceChildren(this.blankOption("自动"));
    configs.forEach((config) => this.datasetMapConfig.appendChild(this.configOption(config)));

    const file = Number(document.getElementById("dataset-file").value);
    const matchingConfig = configs.find((config) => (config.trajectory_files || []).includes(file));
    if (matchingConfig) this.datasetMapConfig.value = matchingConfig.name;
  }

  updateMapConfigSelect() {
    this.mapConfig.replaceChildren(this.blankOption("无"));
    this.options.map_configs.forEach((config) => this.mapConfig.appendChild(this.configOption(config)));
  }

  blankOption(label) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = label;
    return option;
  }

  configOption(config) {
    const option = document.createElement("option");
    option.value = config.name;
    option.textContent = `${config.name} ${config.description || ""}`.trim();
    return option;
  }

  showSource(source) {
    this.source = source;
    this.sourceButtons.forEach((button) => {
      const isActive = button.dataset.sourceTab === source;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    document.querySelectorAll("[data-source-panel]").forEach((panel) => {
      const isHidden = panel.getAttribute("data-source-panel") !== source;
      panel.hidden = isHidden;
      panel.classList.toggle("is-hidden", isHidden);
    });
  }

  datasetPayload() {
    return {
      dataset: this.datasetSelect.value,
      folder: document.getElementById("dataset-folder").value,
      file: document.getElementById("dataset-file").value,
      frames: Number(document.getElementById("dataset-frames").value || 300),
      max_fps: Number(document.getElementById("dataset-max-fps").value || 30),
      perception_range: Number(document.getElementById("dataset-range").value || 80),
      map_config: parseOptionalText(this.datasetMapConfig.value),
      osm_path: parseOptionalText(document.getElementById("dataset-osm").value),
      start_time_ms: parseOptionalNumber(document.getElementById("dataset-start").value),
      follow_id: parseOptionalNumber(document.getElementById("dataset-follow").value),
      ids: parseOptionalText(document.getElementById("dataset-ids").value),
      lanelet2: document.getElementById("dataset-lanelet2").checked,
      loop: document.getElementById("dataset-loop").checked
    };
  }

  mapPayload() {
    return {
      osm_path: document.getElementById("map-osm").value,
      map_config: parseOptionalText(this.mapConfig.value),
      lanelet2: document.getElementById("map-lanelet2").checked
    };
  }

  async startDatasetPreview() {
    try {
      this.setStatus("加载中", "running");
      await this.request("/api/preview/dataset", this.datasetPayload());
      await this.refreshStatus();
      window.setTimeout(() => this.refreshStatus(), 250);
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async loadMapPreview() {
    try {
      this.setStatus("加载地图", "running");
      const result = await this.request("/api/preview/map", this.mapPayload());
      this.applyStatus(result);
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async startLive() {
    try {
      this.setStatus("等待实时帧", "running");
      await this.request("/api/preview/live", {});
      await this.refreshStatus();
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async loadRecordings() {
    try {
      const data = await this.request("/api/recordings");
      this.replaySelect.replaceChildren();
      (data.recordings || []).forEach((recording) => {
        const option = document.createElement("option");
        option.value = recording.name;
        option.textContent = recording.name;
        this.replaySelect.appendChild(option);
      });
      if (!this.replaySelect.options.length) {
        this.replaySelect.appendChild(this.blankOption("暂无录制"));
      }
      if (data.recording) {
        this.isRecordingFrames = true;
        this.recordButton.textContent = `停止录制（${data.recording}）`;
        this.recordButton.classList.add("is-recording");
      }
    } catch (error) {
      // Recording list is non-critical; leave the placeholder option.
    }
  }

  async toggleFrameRecording() {
    try {
      if (this.isRecordingFrames) {
        const result = await this.request("/api/record/stop", {});
        this.isRecordingFrames = false;
        this.recordButton.textContent = "开始录制";
        this.recordButton.classList.remove("is-recording");
        this.setStatus(result.message || "已保存", "complete");
        await this.loadRecordings();
      } else {
        const name = parseOptionalText(document.getElementById("record-name").value);
        const result = await this.request("/api/record/start", name ? { name } : {});
        this.isRecordingFrames = true;
        this.recordButton.textContent = `停止录制（${result.name}）`;
        this.recordButton.classList.add("is-recording");
        this.setStatus("录制中", "running");
      }
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async startReplay() {
    const name = this.replaySelect.value;
    if (!name) return;
    try {
      this.setStatus("加载录制", "running");
      await this.request("/api/preview/replay", { name, max_fps: 30 });
      await this.refreshStatus();
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async stopPreview() {
    try {
      const result = await this.request("/api/preview/stop", {});
      this.applyStatus(result);
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async togglePause() {
    try {
      const result = await this.request(
        this.isPaused ? "/api/preview/resume" : "/api/preview/pause",
        {}
      );
      this.applyStatus(result);
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  async refreshStatus() {
    try {
      const status = await this.request("/api/preview/status");
      this.applyStatus(status);
    } catch {
      this.setStatus("未连接", "error");
    }
  }

  applyStatus(status) {
    const progress = Number(status.progress || 0);
    this.progress.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
    const hasReplayProgress = status.source === "dataset" && Number(status.total_frames || 0) > 0;
    this.progressTrack.classList.toggle("is-hidden", !hasReplayProgress);
    this.setStreamMode(status, hasReplayProgress);
    this.isPaused = status.status === "paused" || status.paused === true;
    this.pauseButton.textContent = this.isPaused ? "继续" : "暂停";

    if (status.status === "running" && status.source === "dataset") {
      const position =
        status.total_frames && status.frame_index
          ? `${status.frame_index}/${status.total_frames}`
          : status.frame || "";
      this.setStatus(
        `${status.sensor_id || "数据集"} ${position} 发送 ${status.sent_frames || 0}`,
        "running"
      );
      return;
    }
    if (status.status === "paused") {
      this.setStatus("已暂停", "idle");
      return;
    }
    if (status.status === "running") {
      const isWaitingLive =
        status.source === "live" && Number(status.sensor_count || 0) === 0 && status.frame == null;
      this.setStatus(isWaitingLive ? "等待实时帧" : status.source === "live" ? "实时输出中" : "运行中", "running");
      return;
    }
    if (status.status === "loading") {
      this.setStatus("加载中", "running");
      return;
    }
    if (status.status === "complete") {
      const frames = status.sent_frames ? ` 发送 ${status.sent_frames}` : "";
      this.setStatus(`${status.sensor_id || "完成"}${frames}`, "idle");
      return;
    }
    if (status.status === "error") {
      this.setStatus(status.message || "错误", "error");
      return;
    }
    if (status.status === "stopped") {
      this.setStatus("已停止", "idle");
      return;
    }
    this.setStatus("就绪", "idle");
  }

  setStreamMode(status, hasReplayProgress) {
    this.streamMode.classList.toggle("is-live", false);
    this.streamMode.classList.toggle("is-replay", false);
    if (hasReplayProgress) {
      this.streamMode.textContent = "回放";
      this.streamMode.classList.toggle("is-replay", true);
      return;
    }
    if (status.status === "running" || status.status === "paused") {
      this.streamMode.textContent = "LIVE";
      this.streamMode.classList.toggle("is-live", true);
      return;
    }
    this.streamMode.textContent = "就绪";
  }

  setStatus(message, mode = "idle") {
    this.status.textContent = message;
    this.status.classList.toggle("is-error", mode === "error");
    this.status.classList.toggle("is-running", mode === "running");
  }
}

class RenderManager {
  constructor() {
    this.container = document.getElementById("sensor-grid");
    this.status = document.getElementById("connection-status");
    this.fpsElement = document.getElementById("render-fps");
    this.frameElement = document.getElementById("render-frame");
    this.sensorCountElement = document.getElementById("render-sensors");
    this.droppedElement = document.getElementById("render-dropped");
    this.sensors = new Map();
    this.layout = "grid";
    this.userLayout = null;
    this.pendingFrame = null;
    this.frameScheduled = false;
    this.renderedFrames = 0;
    this.fpsFrames = 0;
    this.lastFpsAt = performance.now();
    this.browserDroppedFrames = 0;
    this.connect();
    this.bindLayoutButtons();
    this.updateStats("-");
  }

  bindLayoutButtons() {
    document.querySelectorAll("[data-layout]").forEach((button) => {
      button.addEventListener("click", () => {
        this.userLayout = button.dataset.layout;
        this.setLayout(button.dataset.layout);
        if (this.socket?.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: "layout.set", layout: this.layout }));
        }
      });
    });
  }

  setLayout(layout) {
    this.layout = layout === "master" || layout === "hierarchical" ? "master" : "grid";
    this.container.classList.toggle("is-master", this.layout === "master");
    document.querySelectorAll("[data-layout]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.layout === this.layout);
    });
    this.sensors.forEach((sensor) => sensor.resize());
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
    this.socket.addEventListener("open", () => this.setConnected(true));
    this.socket.addEventListener("close", () => {
      this.setConnected(false);
      window.setTimeout(() => this.connect(), 1000);
    });
    this.socket.addEventListener("message", (event) => this.handleMessage(event));
  }

  setConnected(isConnected) {
    this.status.classList.toggle("is-connected", isConnected);
    this.status.title = isConnected ? "Connected" : "Disconnected";
  }

  handleMessage(event) {
    const message = JSON.parse(event.data);
    if (message.type === "layout.set") {
      this.userLayout = message.layout;
      this.setLayout(message.layout);
    }
    if (message.type === "frame.update") this.queueFrame(message);
  }

  queueFrame(message) {
    if (this.pendingFrame) this.browserDroppedFrames += 1;
    this.pendingFrame = message;
    if (this.frameScheduled) return;

    this.frameScheduled = true;
    window.requestAnimationFrame(() => this.renderPendingFrame());
  }

  renderPendingFrame() {
    this.frameScheduled = false;
    const message = this.pendingFrame;
    this.pendingFrame = null;
    if (!message) return;

    this.updateFrame(message.payload, message.frame_id);
    this.recordRenderedFrame(message.frame_id);
  }

  updateFrame(payload, frameId) {
    // Frame payloads carry a default layout; an explicit user/API choice wins over it.
    if (payload.layout && !this.userLayout) this.setLayout(payload.layout);
    (payload.sensor_id_to_remove || []).forEach((sensorId) => this.removeSensor(sensorId));

    const activeSensorIds = new Set();
    (payload.sensors || []).forEach((sensor) => {
      activeSensorIds.add(sensor.id);
      let sensorView = this.sensors.get(sensor.id);
      if (!sensorView) {
        sensorView = new SensorView(sensor);
        this.sensors.set(sensor.id, sensorView);
        this.container.appendChild(sensorView.element);
        sensorView.resize();
      }
      sensorView.update(sensor);
    });

    if (payload.remove_missing_sensors !== false) {
      Array.from(this.sensors.keys()).forEach((sensorId) => {
        if (!activeSensorIds.has(sensorId)) this.removeSensor(sensorId);
      });
    }

    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "render.ack", frame_id: frameId }));
    }
  }

  recordRenderedFrame(frameId) {
    const now = performance.now();
    this.renderedFrames += 1;
    this.fpsFrames += 1;
    if (now - this.lastFpsAt >= 500) {
      const fps = Math.round((this.fpsFrames * 1000) / (now - this.lastFpsAt));
      this.fpsFrames = 0;
      this.lastFpsAt = now;
      this.updateStats(frameId, fps);
      return;
    }

    this.updateStats(frameId);
  }

  updateStats(frameId, fps = null) {
    if (fps !== null && this.fpsElement) this.fpsElement.textContent = String(fps);
    if (this.frameElement) this.frameElement.textContent = String(frameId ?? "-");
    if (this.sensorCountElement) this.sensorCountElement.textContent = String(this.sensors.size);
    if (this.droppedElement) this.droppedElement.textContent = String(this.browserDroppedFrames);
  }

  removeSensor(sensorId) {
    const sensorView = this.sensors.get(sensorId);
    if (!sensorView) return;
    sensorView.dispose();
    this.sensors.delete(sensorId);
  }
}

class ScreenRecorder {
  constructor(renderManager) {
    this.renderManager = renderManager;
    this.button = document.getElementById("record-screen");
    this.recording = false;
    if (!window.MediaRecorder) {
      this.button.disabled = true;
      this.button.title = "当前浏览器不支持 MediaRecorder";
      return;
    }
    this.button.addEventListener("click", () => (this.recording ? this.stop() : this.start()));
  }

  supportedMimeType() {
    // Prefer MP4 (H.264) for out-of-the-box player compatibility; fall back to WebM.
    const types = [
      "video/mp4;codecs=avc1.42E01E",
      "video/mp4",
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ];
    return types.find((type) => MediaRecorder.isTypeSupported(type)) || null;
  }

  start() {
    const rect = this.renderManager.container.getBoundingClientRect();
    const mimeType = this.supportedMimeType();
    if (!mimeType || !rect.width || !rect.height) return;

    this.canvas = document.createElement("canvas");
    this.canvas.width = Math.round(rect.width);
    this.canvas.height = Math.round(rect.height);
    this.context = this.canvas.getContext("2d");
    this.chunks = [];
    this.stream = this.canvas.captureStream(30);
    this.recorder = new MediaRecorder(this.stream, { mimeType });
    this.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) this.chunks.push(event.data);
    });
    this.recorder.addEventListener("stop", () => this.download());
    this.recorder.start(1000);
    this.recording = true;
    this.button.textContent = "停止录屏";
    this.button.classList.add("is-recording");
    this.compose();
  }

  compose() {
    if (!this.recording) return;
    const gridRect = this.renderManager.container.getBoundingClientRect();
    this.context.fillStyle = "#05070a";
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.renderManager.sensors.forEach((sensorView) => {
      const canvas = sensorView.renderer.domElement;
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      this.context.drawImage(
        canvas,
        rect.left - gridRect.left,
        rect.top - gridRect.top,
        rect.width,
        rect.height
      );
    });
    window.requestAnimationFrame(() => this.compose());
  }

  stop() {
    this.recording = false;
    this.button.textContent = "开始录屏";
    this.button.classList.remove("is-recording");
    this.recorder.stop();
    this.stream.getTracks().forEach((track) => track.stop());
  }

  download() {
    const blob = new Blob(this.chunks, { type: this.recorder.mimeType });
    const extension = this.recorder.mimeType.includes("mp4") ? "mp4" : "webm";
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `tactics2d-${stamp}.${extension}`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.tactics2dFrontend = new RenderManager();
  window.tactics2dPreviewControls = new PreviewControls();
  window.tactics2dScreenRecorder = new ScreenRecorder(window.tactics2dFrontend);
});
