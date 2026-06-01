import * as THREE from "three";

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
    this.roadObjects = new Map();
    this.participantObjects = new Map();

    this.element = document.createElement("section");
    this.element.className = "sensor-view";
    this.element.id = sensor.id;

    this.label = document.createElement("div");
    this.label.className = "sensor-label";
    this.label.textContent = sensor.id;
    this.element.appendChild(this.label);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.hole);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.addEventListener("webglcontextrestored", () => this.resize());
    this.element.appendChild(this.renderer.domElement);

    this.camera = new THREE.OrthographicCamera(-50, 50, 50, -50, 0.1, 1000);
    this.camera.position.set(0, 0, 200);
    this.camera.lookAt(0, 0, 0);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.element);
    this.updateView(sensor);
  }

  resize() {
    const width = Math.max(1, this.element.clientWidth);
    const height = Math.max(1, this.element.clientHeight);
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
    const width = Math.max(1, this.element.clientWidth);
    const height = Math.max(1, this.element.clientHeight);
    const aspect = width / height;
    const range = this.perceptionRange;
    this.camera.left = -range * aspect;
    this.camera.right = range * aspect;
    this.camera.top = range;
    this.camera.bottom = -range;
    this.camera.far = Math.max(1000, range * 8);
    this.camera.updateProjectionMatrix();
  }

  updateView(sensor) {
    const position = sensor.position || [0, 0];
    this.perceptionRange = sensor.perception_range || this.perceptionRange;
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
    this.sourceSelect = document.getElementById("preview-source");
    this.datasetForm = document.getElementById("dataset-form");
    this.mapForm = document.getElementById("map-form");
    this.demoPanel = document.getElementById("demo-panel");
    this.datasetSelect = document.getElementById("dataset-select");
    this.datasetMapConfig = document.getElementById("dataset-map-config");
    this.mapConfig = document.getElementById("map-config");
    this.status = document.getElementById("preview-status");
    this.options = { levelx_datasets: [], map_configs: [], defaults: {} };

    this.bind();
    this.loadOptions();
    window.setInterval(() => this.refreshStatus(), 1000);
  }

  bind() {
    this.sourceSelect.addEventListener("change", () => this.showSource(this.sourceSelect.value));
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
    document.getElementById("demo-button").addEventListener("click", () => this.startDemo());
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
      this.showSource(this.sourceSelect.value);
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
    document.querySelectorAll("[data-source-panel]").forEach((panel) => {
      panel.classList.toggle("is-hidden", panel.getAttribute("data-source-panel") !== source);
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
      lanelet2: document.getElementById("dataset-lanelet2").checked
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

  async startDemo() {
    try {
      this.setStatus("播放示例", "running");
      await this.request("/api/preview/demo", {
        max_fps: Number(document.getElementById("demo-max-fps").value || 30)
      });
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

  async refreshStatus() {
    try {
      const status = await this.request("/api/preview/status");
      this.applyStatus(status);
    } catch {
      this.setStatus("未连接", "error");
    }
  }

  applyStatus(status) {
    if (status.status === "running" && status.source === "dataset") {
      this.setStatus(
        `${status.sensor_id || "数据集"} ${status.frame || ""} 发送 ${status.sent_frames || 0}`,
        "running"
      );
      return;
    }
    if (status.status === "running") {
      this.setStatus(status.source === "demo" ? "示例播放中" : "运行中", "running");
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
    this.sensors = new Map();
    this.layout = "grid";
    this.connect();
    this.bindLayoutButtons();
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
    if (message.type === "frame.update") this.updateFrame(message.payload, message.frame_id);
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
