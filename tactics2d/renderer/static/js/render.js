console.log("render.js loaded");

const socket = io("http://127.0.0.1:5000");

const scene = new THREE.Scene();
const aspect = window.innerWidth / window.innerHeight;
const d = 100;
const camera = new THREE.OrthographicCamera(
    -d * aspect, d * aspect,
    d, -d,
    1, 1000
);

camera.position.set(200, 0, 100);
camera.lookAt(200, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(0, 100, 100);
scene.add(directionalLight);

const roadObjects = new Map();
const participantObjects = new Map();


window.addEventListener('resize', () => {
    const aspect = window.innerWidth / window.innerHeight;
    camera.left = -d * aspect;
    camera.right = d * aspect;
    camera.top = d;
    camera.bottom = -d;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});


function parseColor(colorStr) {
    try {
        return new THREE.Color(colorStr);
    } catch (e) {
        console.warn("Invalid color:", colorStr);
        return new THREE.Color(0xff00ff);
    }
}


function createCircle(element) {
    const geometry = new THREE.CircleGeometry(element.radius, 32);
    const material = new THREE.MeshBasicMaterial({ color: parseColor(element.color) });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = element.order;
    return mesh;
}


function updateCircle(element) {
    const obj = participantObjects.get(element.id);
    if (obj) {
        obj.position.set(element.position[0], element.position[1], 0);
    }
}


function createPolygon(element) {
    const shape = new THREE.Shape();
    element.geometry.forEach((pt, i) => {
        if (i === 0) shape.moveTo(pt[0], pt[1]);
        else shape.lineTo(pt[0], pt[1]);
    });

    const geometry = new THREE.ShapeGeometry(shape);
    const material = new THREE.MeshBasicMaterial({
        color: parseColor(element.color),
        side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = element.order || 0;
    return mesh;
}


function updatePolygon(element) {
    const obj = participantObjects.get(element.id);
    if (obj) {
        obj.position.set(element.position[0], element.position[1], 0);
        obj.rotation.set(0, 0, element.rotation);
    }
}


function createLine(element) {
    const points = element.geometry.map(([x, y]) => new THREE.Vector3(x, y, 0));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const dashed = element.type.includes("dashed");

    let material;
    if (dashed) {
        material = new THREE.LineDashedMaterial({
            color: parseColor(element.color),
            dashSize: 3,
            gapSize: 2,
            linewidth: element.lineWidth,
        });
    } else {
        material = new THREE.LineBasicMaterial({
            color: parseColor(element.color),
            linewidth: element.lineWidth
        });
    }

    const line = new THREE.Line(geometry, material);
    if (dashed) line.computeLineDistances();

    line.renderOrder = element.order || 0;

    return line;
}


function updateRoadElements(roadData) {
    if (roadData.road_id_to_remove) {
        const obj = roadObjects.get(roadData.road_id_to_remove);
        if (obj) {
            scene.remove(obj);
            roadObjects.delete(roadData.road_id_to_remove);
        }
    }

    roadData.road_element.forEach(element => {
        let mesh;
        if (element.type === "polygon") {
            mesh = createPolygon(element);
        } else if (element.type === "line" || element.type === "dashed_line") {
            mesh = createLine(element);
        }

        if (mesh) {
            scene.add(mesh);
            roadObjects.set(element.id, mesh);
        }
    });
}


// 修复后的participant更新逻辑
function updateParticipants(participantData) {
    const newParticipants = new Set(participantData.participant_id_to_create || []);
    const toRemove = new Set(participantData.participant_id_to_remove || []);

    // 移除对象
    toRemove.forEach(id => {
        const obj = participantObjects.get(id);
        if (obj) {
            scene.remove(obj);
            participantObjects.delete(id);
        }
    });

    // 更新/创建对象
    (participantData.participants || []).forEach(participant => {
        // 更新现有对象
        const existing = participantObjects.get(participant.id);
        if (existing) {
            if (participant.type === "polygon") {
                existing.position.set(participant.position[0], participant.position[1], 0);
                existing.rotation.set(0, 0, participant.rotation || 0);
                existing.renderOrder = participant.order || 0;
            } else if (participant.type === "circle") {
                existing.position.set(participant.position[0], participant.position[1], 0);
                existing.scale.set(participant.radius, participant.radius, 1);
                existing.renderOrder = participant.order || 0;
            }
        }
        // 创建新对象
        else if (newParticipants.has(participant.id)) {
            let mesh;
            if (participant.type === "polygon") {
                mesh = createPolygon(participant);
            } else if (participant.type === "circle") {
                mesh = createCircle(participant);
            }

            if (mesh) {
                participantObjects.set(participant.id, mesh);
                scene.add(mesh);
            }
        }
    });
}

function clearScene() {
    while (scene.children.length > 0) {
        scene.remove(scene.children[0]);
    }
}

socket.on('connect', () => {
    console.log('Socket.IO connected. Socket ID:', socket.id);
    clearScene();
});


socket.on("geometry_data", function(data) {
    console.log("Rendering frame:", data.frame);

    // Update road elements
    if (data.map_data) {
        updateRoadElements(data.map_data);
    }

    // Update participants
    if (data.participant_data) {
        updateParticipants(data.participant_data);
    }
});

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
