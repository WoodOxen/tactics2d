console.log("✅ render.js loaded");

const socket = io("http://127.0.0.1:5000");

// 场景、相机、渲染器初始化
const scene = new THREE.Scene();
const aspect = window.innerWidth / window.innerHeight;
const d = 100; // 可视范围的一半大小（你可以按需调整）
const camera = new THREE.OrthographicCamera(
    -d * aspect, d * aspect,
    d, -d,
    1, 1000
);

camera.position.set(100, 200, 0); // X=200, Y=高处, Z=0
camera.up.set(0, 0, -1);        // 设置 Z 轴向上
camera.lookAt(100, 0, 0);         // 看向中心点

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 添加光源
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(0, 100, 100);
scene.add(directionalLight);


// 工具函数：颜色转换
function parseColor(colorStr) {
    try {
        return new THREE.Color(colorStr);
    } catch (e) {
        console.warn("Invalid color:", colorStr);
        return new THREE.Color(0xff00ff); // fallback pink
    }
}

// 工具函数：创建圆形
function createCircle(center, radius, color, lineWidth) {
    const geometry = new THREE.CircleGeometry(radius, 32);
    const material = new THREE.MeshBasicMaterial({ color: parseColor(color) });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(center[0], 0, center[1]); // Y 为高度
    return mesh;
}

// 工具函数：创建多边形
function createPolygon(coords, color) {
    const shape = new THREE.Shape();
    coords.forEach((pt, i) => {
        if (i === 0) shape.moveTo(pt[0], pt[1]);
        else shape.lineTo(pt[0], pt[1]);
    });

    const geometry = new THREE.ShapeGeometry(shape);
    const material = new THREE.MeshBasicMaterial({ color: parseColor(color) });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2; // 放平
    return mesh;
}

// 工具函数：创建线或虚线
function createLine(coords, color, dashed, lineWidth) {
    const points = coords.map(([x, y]) => new THREE.Vector3(x, 0.1, y));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    let material;
    if (dashed) {
        material = new THREE.LineDashedMaterial({
            color: parseColor(color),
            dashSize: 3,
            gapSize: 2,
            linewidth: lineWidth,
        });
    } else {
        material = new THREE.LineBasicMaterial({
            color: parseColor(color),
            linewidth: lineWidth
        });
    }

    const line = new THREE.Line(geometry, material);
    if (dashed) line.computeLineDistances();

    return line;
}

// 清除旧物体
function clearScene() {
    while (scene.children.length > 0) {
        scene.remove(scene.children[0]);
    }
}

socket.on('connect', () => {
    console.log('✅ Socket.IO connected. Socket ID:', socket.id);
});

// 监听 WebSocket 数据并渲染
socket.on("geometry_data", function(data) {
    console.log("Rendering frame:", data.frame);
    clearScene();

    data.geometry.forEach(obj => {
        let mesh;
        if (obj.type === "circle") {
            mesh = createCircle(obj.center, obj.radius, obj.color, obj.lineWidth);
        } else if (obj.type === "polygon") {
            mesh = createPolygon(obj.geometry, obj.color);
        } else if (obj.type === "line" || obj.type === "dashedline") {
            mesh = createLine(obj.geometry, obj.color, obj.type === "dashedline", obj.lineWidth);
        }

        if (mesh) scene.add(mesh);
    });
});

// 渲染循环
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
