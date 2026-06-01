import * as THREE from "three";

const DEFAULT_VIEWPORT_ASPECT = 16 / 9;

const COLORS = {
  area: "#2f3542",
  bicycle: "#fd9644",
  bus: "#4b6584",
  cyclist: "#fd9644",
  default: "#a5b1c2",
  heading_arrow: "#101418",
  hole: "#f4f7f7",
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
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
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
    this.source = "dataset";
    this.sourceButtons = Array.from(document.querySelectorAll("[data-source-tab]"));
    this.datasetForm = document.getElementById("dataset-form");
    this.mapForm = document.getElementById("map-form");
    this.livePanel = document.getElementById("live-panel");
    this.datasetSelect = document.getElementById("dataset-select");
    this.datasetMapConfig = document.getElementById("dataset-map-config");
    this.mapConfig = document.getElementById("map-config");
    this.status = document.getElementById("preview-status");
    this.progress = document.getElementById("preview-progress");
    this.progressTrack = document.getElementById("progress-track");
    this.streamMode = document.getElementById("stream-mode");
    this.pauseButton = document.getElementById("pause-preview");
    this.isPaused = false;
    this.options = { levelx_datasets: [], map_configs: [], defaults: {} };

    this.bind();
    this.loadOptions();
    window.setInterval(() => this.refreshStatus(), 500);
  }

  bind() {
    this.sourceButtons.forEach((button) => {
      button.addEventListener("click", () => this.showSource(button.dataset.sourceTab));
    });
    this.datasetSelect.addEventListener("change", () => this.updateDatasetMapConfigs());
    document
      .getElementById("dataset-file")
      .addEventListener("change", () => this.updateDatasetMapConfigs());
    this.datasetForm.addEventListener("submit", (event) => {
      event.preventDefault();
      this.startDatasetPreview();
    });
    this.mapForm.addEventListener("submit", (event) => {
      event.preventDefault();
      this.loadMapPreview();
    });
    document.getElementById("live-button").addEventListener("click", () => this.startLive());
    this.pauseButton.addEventListener("click", () => this.togglePause());
    document.getElementById("stop-preview").addEventListener("click", () => this.stopPreview());
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
      this.updateDatasetMapConfigs();
      this.updateMapConfigSelect();
      this.showSource(this.source);
    } catch (error) {
      this.setStatus(error.message, "error");
    }
  }

  populateDatasetSelect() {
    this.datasetSelect.replaceChildren();
    this.options.levelx_datasets.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset;
      option.textContent = dataset;
      this.datasetSelect.appendChild(option);
    });
  }

  populateDefaults() {
    const defaults = this.options.defaults || {};
    this.datasetSelect.value = defaults.dataset || "highD";
    document.getElementById("dataset-folder").value = defaults.folder || "";
    document.getElementById("dataset-file").value = defaults.file || "";
    document.getElementById("dataset-frames").value = defaults.frames || 300;
    document.getElementById("dataset-max-fps").value = defaults.max_fps || 30;
    document.getElementById("dataset-range").value = defaults.perception_range || 80;
    document.getElementById("map-osm").value = "data/highD_map/highD_1.osm";
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
      this.setStatus(status.source === "live" ? "实时输出中" : "运行中", "running");
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
    if (message.type === "layout.set") this.setLayout(message.layout);
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
    if (payload.layout) this.setLayout(payload.layout);
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

window.addEventListener("DOMContentLoaded", () => {
  window.tactics2dFrontend = new RenderManager();
  window.tactics2dPreviewControls = new PreviewControls();
});
