console.log("renderManager.js loaded");

let renderManagerInstance = null;

window.addEventListener("DOMContentLoaded", () => {
    if (!renderManagerInstance) {
        renderManagerInstance = new RenderManager();
    }
});

class windowContainer {
    constructor(containerId, perceptionRange, initialPosition = [0, 0], initialYaw = 0) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();

        this.perceptionRange = perceptionRange;

        this.roadObjects = new Map();
        this.participantObjects = new Map();

        this.initRenderer();
        this.initCamera();
        this.initLight();

        this.updateView(initialPosition, initialYaw);
    }

    // Initialize the WebGL renderer
    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(
            this.container.clientWidth,
            this.container.clientHeight
        );
        this.container.appendChild(this.renderer.domElement);
    }

    // Initialize the orthographic camera
    initCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;

        this.camera = new THREE.OrthographicCamera(
            - this.perceptionRange * aspect, this.perceptionRange * aspect,
            this.perceptionRange, -this.perceptionRange, 1, 1000
        )
    }

    // Initialize the lighting for the scene
    initLight() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 100, 100);
        this.scene.add(directionalLight);
    }

    // Update the renderer size based on the container dimensions
    updateRenderer(width, height) {
        this.renderer.setSize(width, height);

        if (this.lastPosition && this.lastYaw !== undefined) {
            this.updateView(this.lastPosition, this.lastYaw);
        }
    }

    // Update the camera view based on the current position and yaw
    updateView(position, yaw) {
        this.lastPosition = position;
        this.lastYaw = yaw;

        const aspect = this.container.clientWidth / this.container.clientHeight;
        const expandFactor = Math.max(
            Math.abs(Math.cos(yaw)) + Math.abs(Math.sin(yaw)) * aspect,
            Math.abs(Math.sin(yaw)) + Math.abs(Math.cos(yaw)) / aspect
        );

        this.camera.left = - this.perceptionRange * expandFactor;
        this.camera.right = this.perceptionRange * expandFactor;
        this.camera.top = this.perceptionRange;
        this.camera.bottom = -this.perceptionRange;
        this.camera.updateProjectionMatrix();

        this.camera.position.set(position[0], position[1], this.perceptionRange * 2);
        this.camera.rotation.z = -yaw;

        this.camera.lookAt(position[0], position[1], 0);
    }

    // Parse color strings and return a THREE.Color object
    parseColor(colorStr) {
        try {
            return new THREE.Color(colorStr);
        } catch (e) {
            console.warn("Invalid color:", colorStr);
            return new THREE.Color(0xeeeeee);
        }
    }

    // Create a circle mesh
    createCircle(element) {
        const geometry = new THREE.CircleGeometry(element.radius || 1, 32);
        const material = new THREE.MeshBasicMaterial({ color: this.parseColor(element.color) });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.renderOrder = element.order || 0;
        return mesh;
    }

    // Create a polygon mesh
    createPolygon(element) {
        const shape = new THREE.Shape();
        element.geometry.forEach((pt, i) => {
            if (i === 0) shape.moveTo(pt[0], pt[1]);
            else shape.lineTo(pt[0], pt[1]);
        });

        const geometry = new THREE.ShapeGeometry(shape);
        const material = new THREE.MeshBasicMaterial({
            color: this.parseColor(element.color),
            side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.renderOrder = element.order || 0;
        return mesh;
    }

    // Create a line mesh
    createLine(element) {
        const points = element.geometry.map(([x, y]) => new THREE.Vector3(x, y, 0));
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const dashed = element.type.includes("dashed");

        let material;
        if (dashed) {
            material = new THREE.LineDashedMaterial({
                color: this.parseColor(element.color),
                dashSize: 3,
                gapSize: 2,
                linewidth: element.line_width,
            });
        } else {
            material = new THREE.LineBasicMaterial({
                color: this.parseColor(element.color),
                linewidth: element.line_width
            });
        }

        const line = new THREE.Line(geometry, material);
        if (dashed) line.computeLineDistances();

        line.renderOrder = element.order || 0;

        return line;
    }

    // Update road elements based on the received data
    updateRoadElements(roadData) {
        if (roadData.road_id_to_remove) {
            const obj = this.roadObjects.get(roadData.road_id_to_remove);
            if (obj) {
                this.scene.remove(obj);
                this.roadObjects.delete(roadData.road_id_to_remove);
            }
        }

        roadData.road_elements.forEach(element => {
            let mesh;
            if (element.type === "polygon") {
                mesh = this.createPolygon(element);
            } else if (element.type.indexOf("line") !== -1) {
                mesh = this.createLine(element);
            }

            if (mesh) {
                this.scene.add(mesh);
                this.roadObjects.set(element.id, mesh);
            }
        });
    }

    // Update participants based on the received data
    updateParticipants(participantData) {
        const newParticipants = new Set(participantData.participant_id_to_create || []);
        const toRemove = new Set(participantData.participant_id_to_remove || []);

        // Remove dead participants
        toRemove.forEach(id => {
            const obj = this.participantObjects.get(id);
            if (obj) {
                this.scene.remove(obj);
                this.participantObjects.delete(id);
            }
        });

        // Update location of living participants
        (participantData.participants || []).forEach(participant => {
            const existing = this.participantObjects.get(participant.id);
            if (existing) {
                if (participant.type === "polygon") {
                    existing.position.set(participant.position[0], participant.position[1], 0);
                    existing.rotation.set(0, 0, participant.rotation || 0);
                } else if (participant.type === "circle") {
                    existing.position.set(participant.position[0], participant.position[1], 0);
                } else console.warn("Unknown participant type:", participant.type);
            }
            else if (newParticipants.has(participant.id)) {
                let mesh;
                if (participant.type === "polygon") {
                    mesh = this.createPolygon(participant);
                } else if (participant.type === "circle") {
                    mesh = this.createCircle(participant);
                } else console.warn("Unknown participant type:", participant.type);

                if (mesh) {
                    this.participantObjects.set(participant.id, mesh);
                    this.scene.add(mesh);
                }
            }
        });
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    clearScene() {
        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }
    }
}


class RenderManager {
    constructor(layout="grid", host="127.0.0.1", port=5000) {
        this.container = document.getElementById("sensor-container-wrapper");
        this.sensors = new Map();
        this.mainSensorId = null;
        this.layout = layout;
        this.animationFrame=null;
        this.socket=null;

        this.host=host;
        this.port=port;

        this.initSocket();

        this.resizeObserver = new ResizeObserver(() => {
            this.updateLayout(this.layout);
        });
        this.resizeObserver.observe(this.container);

        console.log("RenderManager loaded");
    }

    initSocket() {
        this.socket=io("http://127.0.0.1:5000");

        this.socket.on("connect", () => {
            console.log("Connected to the server")
        });

        this.socket.on("connect_error", (err) => {
            console.error("Socket connection error:", err);
        });

        this.socket.on("layout", (layout) => {
            this.updateLayout(layout);
        });

        this.socket.on("sensor_data", (sensor_data) => {
            this.renderAll(sensor_data);
        });

        this.socket.on("disconnect", () => {
            console.log("Disconnected from the server.");
        });
    }

    // Adjust the layout to fit all sensors in a grid.
    updateGridLayout() {
        const numWindow = this.sensors.size;
        if (numWindow === 0) return;

        const numCols = Math.ceil(Math.sqrt(numWindow));
        const numRows = Math.ceil(numWindow / numCols);

        const wrapperHeight = this.container.clientHeight;
        const windowLength = Math.floor(wrapperHeight / numRows / 100) * 100;

        let i = 0;
        this.sensors.forEach((sensor) => {
            const row = Math.floor(i / numCols);
            const col = i % numCols;

            sensor.container.style.width = `${windowLength}px`;
            sensor.container.style.height = `${windowLength}px`;
            sensor.container.style.position = "absolute";
            sensor.container.style.left = `${col * windowLength}px`;
            sensor.container.style.top = `${row * windowLength}px`;

            sensor.updateRenderer(windowLength, windowLength);

            i++;
        });
        console.log("Updated to grid layout.")
    }

    updateHierarchicalLayout() {
        const numWindow = this.sensors.size;
        if (numWindow === 0) return;

        if (this.mainSensorId === null || !this.sensors.has(this.mainSensorId)) {
            this.mainSensorId = Array.from(this.sensors.keys())[0];
        }

        const wrapperHeight = this.container.clientHeight;
        const wrapperWidth = this.container.clientWidth;
        const mainLength = Math.floor(wrapperHeight * 0.7 / 100) * 100;
        const subLength = max(Math.floor(wrapperHeight * 0.3 / 100) * 100, Math.floor(wrapperWidth / numWindow / 100) * 100);

        const mainSensor = this.sensors.get(this.mainSensorId);
        mainSensor.container.style.width = `${mainLength}px`;
        mainSensor.container.style.height = `${mainLength}px`;
        mainSensor.container.style.position = "absolute";
        mainSensor.container.style.left = `${(wrapperWidth - mainLength) / 2}px`;
        mainSensor.container.style.top = `${(wrapperHeight - mainLength) / 2}px`;
        mainSensor.updateRenderer(mainLength, mainLength);

        const sensorIds = Array.from(this.sensors.keys());
        sensorIds.forEach((id, index) => {
            if (id === this.mainSensorId) return;

            const sensor = this.sensors.get(id);
            sensor.container.style.width = `${subLength}px`;
            sensor.container.style.height = `${subLength}px`;
            sensor.container.style.position = "absolute";

            const row = Math.floor(index / 2);
            const col = index % 2;

            sensor.container.style.left = `${(wrapperWidth - subLength * 2) / 2 + col * subLength}px`;
            sensor.container.style.top = `${(wrapperHeight - subLength * 2) / 2 + row * subLength}px`;

            sensor.updateRenderer(subLength, subLength);
        });

        console.log("Updated to hierarchical layout.");
    }

    updateLayout(layout) {
        if (layout === "grid") {
                this.updateGridLayout();
            } else if (layout === "hierarchical") {
                this.updateHierarchicalLayout();
            } else console.warn("Invalid layout type. Reverting to the current layout.");
    }

    updateLayoutSelect() {
        const select = document.getElementById("layout-select");
        if (!select) return;

        select.addEventListener("change", (event) => {
            const newLayout = event.target.value;
            if (newLayout !== this.layout) {
                this.layout = newLayout;
                this.updateLayout(newLayout);
                if (this.socket && this.socket.connected) {
                    this.socket.emit("layout", newLayout);
                }
            }
        });
    }

    renderAll(sensorData) {
        let isSensorChanged = false;

        sensorData.forEach((sensor) => {
            let container = document.getElementById(sensor.id);
            if (!container) {
                container = document.createElement("div");
                container.id = sensor.id;
                container.className = "sensor-window";
                container.style.position = "absolute";
                document.getElementById("sensor-container-wrapper").appendChild(container);
            }

            if (!this.sensors.has(sensor.id)) {
                console.log(`Creating new sensor container for ${sensor.id}`);
                const newSensor = new windowContainer(
                    sensor.id, sensor.perception_range, sensor.position, sensor.yaw
                );
                this.sensors.set(sensor.id, newSensor);
                isSensorChanged = true;
            }

            const sensorObject = this.sensors.get(sensor.id);
            if (sensor.map_data) sensorObject.updateRoadElements(sensor.map_data);
            if (sensor.participant_data) sensorObject.updateParticipants(sensor.participant_data);
            sensorObject.updateView(sensor.position, sensor.yaw);
            sensorObject.render();
        });

        if (isSensorChanged) {
            this.updateLayout(this.layout);
        }

        if (this.socket && this.socket.connected) {
            this.socket.emit("render_complete");
        }
    }
}
