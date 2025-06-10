class WindowCamera {
    constructor(containerId, perceptionRange, initialPosition = [0, 0], initialYaw = 0) {
        // Store container and create renderer elements
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();

        this.perceptionRange = perceptionRange;
        this.aspect = this.container.clientWidth / this.container.clientHeight;

        this.roadObjects = new Map();
        this.participantObjects = new Map();

        this.initRenderer();
        this.initCamera();
        this.initLight();

        this.setView(initialPosition, initialYaw);
    }

    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(
            this.container.clientWidth,
            this.container.clientHeight
        );
        this.container.appendChild(this.renderer.domElement);
    }

    initCamera() {
        this.camera = new THREE.OrthographicCamera(
            - this.perceptionRange * this.aspect, this.perceptionRange * this.aspect,
            this.perceptionRange, -this.perceptionRange, 1, 1000
        )

        this._initialLeft = -this.perceptionRange * this.aspect;
        this._initialRight = this.perceptionRange * this.aspect;
        this._initialTop = this.perceptionRange;
        this._initialBottom = -this.perceptionRange;
    }

    setView(position, yaw) {
        const expandFactor = Math.max(
            Math.abs(Math.cos(yaw)) + Math.abs(Math.sin(yaw)) * this.aspect,
            Math.abs(Math.sin(yaw)) + Math.abs(Math.cos(yaw)) / this.aspect
        );

        this.camera.left = this._initialLeft * expandFactor;
        this.camera.right = this._initialRight * expandFactor;
        this.camera.top = this._initialTop * expandFactor;
        this.camera.bottom = this._initialBottom * expandFactor;
        this.camera.updateProjectionMatrix();

        this.camera.position.set(position[0], position[1], this.perceptionRange * 2);
        this.camera.rotation.z = -yaw;

        this.camera.lookAt(
            position[0],
            position[1],
            0
        );
    }

    initLight() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 100, 100);
        this.scene.add(directionalLight);
    }

    parseColor(colorStr) {
        try {
            return new THREE.Color(colorStr);
        } catch (e) {
            console.warn("Invalid color:", colorStr);
            return new THREE.Color(0xff00ff);
        }
    }

    createCircle(element) {
        const geometry = new THREE.CircleGeometry(element.radius, 32);
        const material = new THREE.MeshBasicMaterial({ color: this.parseColor(element.color) });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.renderOrder = element.order || 0;
        return mesh;
    }

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
                linewidth: element.lineWidth,
            });
        } else {
            material = new THREE.LineBasicMaterial({
                color: this.parseColor(element.color),
                linewidth: element.lineWidth
            });
        }

        const line = new THREE.Line(geometry, material);
        if (dashed) line.computeLineDistances();

        line.renderOrder = element.order || 0;

        return line;
    }

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
                }
            }
            else if (newParticipants.has(participant.id)) {
                let mesh;
                if (participant.type === "polygon") {
                    mesh = this.createPolygon(participant);
                } else if (participant.type === "circle") {
                    mesh = this.createCircle(participant);
                }

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
    constructor() {

    }
}
