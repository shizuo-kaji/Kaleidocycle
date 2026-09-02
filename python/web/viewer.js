(function initialiseStudio() {
  "use strict";

  const THREE = window.THREE;
  const Mathematics = window.Kaleidocycle;
  const staticFallback = window.KALEIDOCYCLE_SAMPLE_FALLBACK;
  const webDirectory = new URL(".", window.location.href);
  const servedFromWebDirectory = /\/web\/$/.test(webDirectory.pathname);
  const dataBaseUrl = new URL(
    servedFromWebDirectory ? "../data/kaleidocycles/" : "./data/kaleidocycles/",
    webDirectory,
  );
  const element = (id) => document.getElementById(id);
  const viewport = element("viewport");

  if (!THREE || !Mathematics) {
    const fatal = element("fatalError");
    fatal.textContent =
      "The local application files are incomplete. Check sim.js and web/vendor/.";
    fatal.classList.add("visible");
    return;
  }

  let samples = {};
  let catalogue = { default: null, samples: [] };
  let sampleSource = "data/";

  const controls = {
    sample: element("sampleSelect"),
    open: element("openFile"),
    export: element("exportFile"),
    file: element("fileInput"),
    flowKind: element("flowKind"),
    flowOrder: element("flowOrder"),
    flowOrderValue: element("flowOrderValue"),
    flowHint: element("flowHint"),
    flowRate: element("flowRate"),
    flowRateValue: element("flowRateValue"),
    orderControl: element("orderControl"),
    play: element("playFlow"),
    reverse: element("reverseFlow"),
    step: element("stepFlow"),
    torsion: element("torsion"),
    torsionValue: element("torsionValue"),
    resetShape: element("resetShape"),
    setInitial: element("setInitial"),
    ribbonWidth: element("ribbonWidth"),
    widthValue: element("widthValue"),
    opacity: element("surfaceOpacity"),
    opacityValue: element("opacityValue"),
    centreline: element("showCentreline"),
    hinges: element("showHinges"),
    handles: element("showHandles"),
    resetView: element("resetView"),
    capture: element("captureView"),
  };

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  if ("outputColorSpace" in renderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewport.insertBefore(renderer.domElement, viewport.firstChild);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 300);
  const modelGroup = new THREE.Group();
  scene.add(modelGroup);

  const hemisphere = new THREE.HemisphereLight(0xd9f3e0, 0x18231d, 2.1);
  scene.add(hemisphere);
  const keyLight = new THREE.DirectionalLight(0xfff4da, 4.2);
  keyLight.position.set(4.5, 7.5, 5.5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.near = 1;
  keyLight.shadow.camera.far = 30;
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x79cda9, 2.3);
  rimLight.position.set(-6, 1.5, -5);
  scene.add(rimLight);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(11, 80),
    new THREE.MeshStandardMaterial({
      color: 0x101713,
      roughness: 0.98,
      metalness: 0,
      transparent: true,
      opacity: 0.52,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -3.25;
  floor.receiveShadow = true;
  scene.add(floor);

  const grid = new THREE.GridHelper(18, 18, 0x314038, 0x202a25);
  grid.position.y = -3.23;
  grid.material.transparent = true;
  grid.material.opacity = 0.28;
  scene.add(grid);

  const surfaceMaterial = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    roughness: 0.31,
    metalness: 0.02,
    clearcoat: 0.36,
    clearcoatRoughness: 0.52,
    transparent: true,
    opacity: Number(controls.opacity.value),
    side: THREE.DoubleSide,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0xdce8dc,
    transparent: true,
    opacity: 0.21,
  });
  const centrelineMaterial = new THREE.LineBasicMaterial({
    color: 0xf0f6ee,
    transparent: true,
    opacity: 0.86,
  });
  const hingeMaterial = new THREE.LineBasicMaterial({
    color: 0xc9f28c,
    transparent: true,
    opacity: 0.68,
  });
  const handleGeometry = new THREE.SphereGeometry(0.105, 20, 14);
  const handleMaterial = new THREE.MeshStandardMaterial({
    color: 0xd8e2d8,
    emissive: 0x1e2b22,
    roughness: 0.36,
    metalness: 0.12,
  });
  const handleHotMaterial = new THREE.MeshStandardMaterial({
    color: 0xd9ff9f,
    emissive: 0x55772d,
    emissiveIntensity: 0.7,
    roughness: 0.28,
  });

  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), surfaceMaterial);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const edges = new THREE.LineSegments(new THREE.BufferGeometry(), edgeMaterial);
  const centreline = new THREE.Line(new THREE.BufferGeometry(), centrelineMaterial);
  const hinges = new THREE.LineSegments(new THREE.BufferGeometry(), hingeMaterial);
  const handleGroup = new THREE.Group();
  modelGroup.add(mesh, edges, centreline, hinges, handleGroup);

  let model;
  let running = false;
  let direction = 1;
  let lastFrameTime = performance.now();
  let projectionCounter = 0;
  let pendingTorsion = null;
  let pendingDragTarget = null;
  let toastTimer;
  let currentFileName = "kaleidocycle";

  const cameraState = {
    yaw: 0.74,
    pitch: 0.34,
    distance: 10,
    target: new THREE.Vector3(),
  };
  const pointerState = {
    mode: null,
    x: 0,
    y: 0,
    vertex: -1,
    hovered: -1,
    plane: new THREE.Plane(),
  };
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  function toast(message) {
    const target = element("toast");
    target.textContent = message;
    target.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => target.classList.remove("visible"), 2400);
  }

  async function loadSampleData() {
    if (window.location.protocol !== "file:") {
      try {
        const catalogueResponse = await fetch(
          new URL("catalog.json", dataBaseUrl),
          { cache: "no-store" },
        );
        if (!catalogueResponse.ok) {
          throw new Error(`catalogue returned ${catalogueResponse.status}`);
        }
        catalogue = await catalogueResponse.json();
        const records = await Promise.all(
          catalogue.samples.map(async (entry) => {
            const response = await fetch(new URL(entry.file, dataBaseUrl));
            if (!response.ok) {
              throw new Error(`${entry.file} returned ${response.status}`);
            }
            return [entry.name, await response.json()];
          }),
        );
        samples = Object.fromEntries(records);
        sampleSource = "data/";
        return;
      } catch (error) {
        console.warn("Could not read the live data catalogue; using fallback.", error);
      }
    }
    if (!staticFallback) {
      throw new Error(
        "No sample catalogue is available. Rebuild web/sample-fallback.js.",
      );
    }
    catalogue = staticFallback.catalogue;
    samples = staticFallback.samples;
    sampleSource = "static fallback";
  }

  function friendlyName(key, data) {
    const n = data.n || data.metadata?.n || "?";
    const family = key.startsWith("generic") ? "Generic" : "Möbius";
    const statedOrientation = data.metadata?.oriented;
    const isOriented =
      statedOrientation === true ||
      String(statedOrientation).toLowerCase() === "true";
    const orientation = isOriented ? "oriented" : "anti-oriented";
    const locus = key.includes("noncritical") ? " · non-critical" : "";
    return `${family} K${n} · ${orientation}${locus}`;
  }

  function populateSamples() {
    controls.sample.replaceChildren();
    catalogue.samples.forEach((entry) => {
      const key = entry.name;
      const data = samples[key];
      const option = document.createElement("option");
      option.value = key;
      option.textContent = friendlyName(key, data);
      controls.sample.append(option);
    });
    controls.sample.value = samples[catalogue.default]
      ? catalogue.default
      : catalogue.samples[0]?.name;
  }

  function threeVector(values) {
    return new THREE.Vector3(values[0], values[1], values[2]);
  }

  function frameBinormal(frame) {
    return new THREE.Vector3(frame[2], frame[5], frame[8]);
  }

  function replaceGeometry(object, geometry) {
    object.geometry.dispose();
    object.geometry = geometry;
  }

  function palette(index, count) {
    const start = new THREE.Color(0x84d6b0);
    const middle = new THREE.Color(0xc9f28c);
    const end = new THREE.Color(0x73a9de);
    const position = count <= 1 ? 0 : index / (count - 1);
    return position < 0.5
      ? start.clone().lerp(middle, position * 2)
      : middle.clone().lerp(end, (position - 0.5) * 2);
  }

  function rebuildGeometry() {
    if (!model) return;
    const positions = [];
    const colours = [];
    const edgePositions = [];
    const hingePositions = [];
    const centred = model.displayVertices();
    const points = centred.map(threeVector);
    const width = Number(controls.ribbonWidth.value);
    const binormals = model.configuration.frames.map(frameBinormal);
    const triangleIndices = [
      [0, 1, 2], [1, 3, 2], [0, 2, 3], [0, 3, 1],
    ];
    const edgeIndices = [
      [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
    ];

    for (let index = 0; index < model.n; index += 1) {
      const vertices = [
        points[index].clone().addScaledVector(binormals[index], width),
        points[index].clone().addScaledVector(binormals[index], -width),
        points[index + 1].clone().addScaledVector(binormals[index + 1], width),
        points[index + 1].clone().addScaledVector(binormals[index + 1], -width),
      ];
      const colour = palette(index, model.n);
      triangleIndices.forEach((triangle) => {
        triangle.forEach((vertexIndex) => {
          positions.push(...vertices[vertexIndex].toArray());
          colours.push(colour.r, colour.g, colour.b);
        });
      });
      edgeIndices.forEach(([a, b]) => {
        edgePositions.push(...vertices[a].toArray(), ...vertices[b].toArray());
      });
      hingePositions.push(
        ...vertices[0].toArray(),
        ...vertices[1].toArray(),
      );
    }

    const surfaceGeometry = new THREE.BufferGeometry();
    surfaceGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    surfaceGeometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(colours, 3),
    );
    surfaceGeometry.computeVertexNormals();
    surfaceGeometry.computeBoundingSphere();
    replaceGeometry(mesh, surfaceGeometry);

    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(edgePositions, 3),
    );
    replaceGeometry(edges, edgeGeometry);

    const hingeGeometry = new THREE.BufferGeometry();
    hingeGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(hingePositions, 3),
    );
    replaceGeometry(hinges, hingeGeometry);

    const centrelineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    replaceGeometry(centreline, centrelineGeometry);

    while (handleGroup.children.length > model.n) {
      handleGroup.remove(handleGroup.children.at(-1));
    }
    while (handleGroup.children.length < model.n) {
      const handle = new THREE.Mesh(handleGeometry, handleMaterial);
      handle.castShadow = true;
      handle.userData.vertex = handleGroup.children.length;
      handleGroup.add(handle);
    }
    handleGroup.children.forEach((handle, index) => {
      handle.position.copy(points[index]);
      handle.material =
        index === pointerState.hovered || index === pointerState.vertex
          ? handleHotMaterial
          : handleMaterial;
    });
    centreline.visible = controls.centreline.checked;
    hinges.visible = controls.hinges.checked;
    handleGroup.visible = controls.handles.checked;
    surfaceMaterial.opacity = Number(controls.opacity.value);
    updateDiagnostics();
  }

  function metric(value) {
    if (!Number.isFinite(value)) return "—";
    return Math.abs(value) < 1e-3 ? value.toExponential(2) : value.toFixed(6);
  }

  function updateDiagnostics() {
    if (!model) return;
    const diagnostics = model.diagnostics();
    element("metricClosure").textContent = metric(diagnostics.closure);
    element("metricMonodromy").textContent = metric(diagnostics.monodromy);
    element("metricE1").textContent = metric(diagnostics.hamiltonian1);
    element("metricE2").textContent = metric(diagnostics.hamiltonian2);
    element("closureStatus").textContent = diagnostics.closure.toExponential(1);
    const closurePill = element("closurePill");
    closurePill.classList.toggle("warn", diagnostics.closure >= 1e-6);
    closurePill.classList.toggle("bad", diagnostics.closure >= 1e-3);
    element("flowTime").textContent = `t = ${model.flowTime.toExponential(2)}`;
    controls.torsionValue.textContent = `${model.torsionAngle.toFixed(4)} rad`;
  }

  function updateCamera() {
    const { yaw, pitch, distance, target } = cameraState;
    const cosine = Math.cos(pitch);
    camera.position.set(
      target.x + distance * cosine * Math.sin(yaw),
      target.y + distance * Math.sin(pitch),
      target.z + distance * cosine * Math.cos(yaw),
    );
    camera.lookAt(target);
  }

  function preferredViewDirection(vertices) {
    const points = vertices.map(threeVector);
    const areaNormal = new THREE.Vector3();
    for (let index = 0; index + 1 < points.length; index += 1) {
      areaNormal.add(
        new THREE.Vector3().crossVectors(points[index], points[index + 1]),
      );
    }
    if (areaNormal.lengthSq() < 1e-10) {
      return new THREE.Vector3(0.63, 0.33, 0.68).normalize();
    }
    areaNormal.normalize();
    const tangent = new THREE.Vector3();
    for (let index = 0; index + 1 < points.length; index += 1) {
      tangent.copy(points[index + 1]).sub(points[index]);
      tangent.addScaledVector(areaNormal, -tangent.dot(areaNormal));
      if (tangent.lengthSq() > 1e-10) break;
    }
    tangent.normalize();
    const side = new THREE.Vector3().crossVectors(areaNormal, tangent).normalize();
    const direction = areaNormal
      .multiplyScalar(0.86)
      .addScaledVector(tangent, 0.38)
      .addScaledVector(side, 0.18)
      .normalize();
    if (direction.y < 0) {
      direction.multiplyScalar(-1);
    }
    direction.y += 0.16;
    direction.normalize();
    return direction;
  }

  function resetCamera() {
    const centred = model ? model.displayVertices() : [[0, 0, 0]];
    const radius = Math.max(2, ...centred.map((point) => Math.hypot(...point)));
    const direction = preferredViewDirection(centred);
    cameraState.yaw = Math.atan2(direction.x, direction.z);
    cameraState.pitch = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1));
    cameraState.distance = Math.max(7.5, radius * 3.65);
    cameraState.target.set(0, 0, 0);
    floor.position.y = -Math.max(3.25, radius * 1.08);
    grid.position.y = floor.position.y + 0.02;
    updateCamera();
  }

  function setRunning(next) {
    running = Boolean(next);
    controls.play.textContent = running ? "Pause flow" : "Run flow";
    controls.play.classList.toggle("running", running);
    element("flowStatus").textContent = running
      ? `${direction < 0 ? "−" : "+"} ${flowLabel()}`
      : "Paused";
    element("flowPill").classList.toggle("warn", running);
  }

  function flowLabel() {
    return controls.flowKind.value === "sine-gordon"
      ? "sine–Gordon"
      : `mKdV X(${controls.flowOrder.value})`;
  }

  function updateFlowControls() {
    const hierarchy = controls.flowKind.value === "hierarchy";
    controls.orderControl.hidden = !hierarchy;
    controls.flowOrder.disabled = !hierarchy;
    controls.orderControl.setAttribute("aria-hidden", String(!hierarchy));
    controls.flowOrderValue.textContent = `X(${controls.flowOrder.value})`;
    controls.flowHint.innerHTML = hierarchy
      ? "All orders up to the finite Cayley–Hamilton bound <em>N</em> are generated from the same Floquet hierarchy as the Python implementation."
      : "The semi-discrete sine–Gordon flow is a separate negative flow; hierarchy order does not apply.";
    controls.flowRateValue.textContent = `${Number(controls.flowRate.value).toFixed(2)}×`;
    element("flowStatus").textContent = running
      ? `${direction < 0 ? "−" : "+"} ${flowLabel()}`
      : "Paused";
  }

  function loadData(data, name, sampleKey = "") {
    try {
      setRunning(false);
      model = new Mathematics.Model(data);
      currentFileName = name.replace(/\.json$/i, "") || model.name;
      controls.flowOrder.max = String(model.n);
      controls.flowOrder.value = String(
        Math.min(Number(controls.flowOrder.value), model.n),
      );
      const sineGordonOption = controls.flowKind.querySelector(
        'option[value="sine-gordon"]',
      );
      sineGordonOption.disabled = model.sign !== -1;
      if (model.sign !== -1 && controls.flowKind.value === "sine-gordon") {
        controls.flowKind.value = "hierarchy";
      }
      const torsionRadius = 0.5;
      controls.torsion.min = String(Math.max(0.08, model.torsionAngle - torsionRadius));
      controls.torsion.max = String(
        Math.min(Math.PI - 0.08, model.torsionAngle + torsionRadius),
      );
      controls.torsion.value = String(model.torsionAngle);
      controls.sample.value = sampleKey;
      element("configurationTag").textContent = `N = ${model.n} · ${sampleSource}`;
      element("cycleStatus").textContent =
        `K${model.n} · ${model.sign === 1 ? "oriented" : "anti-oriented"}`;
      element("editState").textContent = "Ready";
      pointerState.vertex = -1;
      pointerState.hovered = -1;
      updateFlowControls();
      rebuildGeometry();
      resetCamera();
    } catch (error) {
      console.error(error);
      toast(error instanceof Error ? error.message : String(error));
    }
  }

  function pointerCoordinates(event) {
    const rectangle = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rectangle.left) / rectangle.width) * 2 - 1;
    pointer.y = -((event.clientY - rectangle.top) / rectangle.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
  }

  function pickHandle(event) {
    if (!controls.handles.checked) return null;
    pointerCoordinates(event);
    return raycaster.intersectObjects(handleGroup.children, false)[0] || null;
  }

  renderer.domElement.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const hit = pickHandle(event);
    renderer.domElement.setPointerCapture(event.pointerId);
    pointerState.x = event.clientX;
    pointerState.y = event.clientY;
    if (hit) {
      setRunning(false);
      pointerState.mode = "vertex";
      pointerState.vertex = hit.object.userData.vertex;
      const normal = camera.getWorldDirection(new THREE.Vector3());
      pointerState.plane.setFromNormalAndCoplanarPoint(normal, hit.object.position);
      renderer.domElement.style.cursor = "grabbing";
      element("editState").textContent = `Moving vertex ${pointerState.vertex}`;
      rebuildGeometry();
    } else {
      pointerState.mode = "orbit";
      renderer.domElement.style.cursor = "grabbing";
    }
  });

  renderer.domElement.addEventListener("pointermove", (event) => {
    if (pointerState.mode === "orbit") {
      const dx = event.clientX - pointerState.x;
      const dy = event.clientY - pointerState.y;
      pointerState.x = event.clientX;
      pointerState.y = event.clientY;
      cameraState.yaw -= dx * 0.007;
      cameraState.pitch = Math.max(
        -1.35,
        Math.min(1.35, cameraState.pitch + dy * 0.006),
      );
      updateCamera();
      return;
    }
    if (pointerState.mode === "vertex") {
      pointerCoordinates(event);
      const target = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(pointerState.plane, target)) {
        pendingDragTarget = target.toArray();
      }
      return;
    }
    const hit = pickHandle(event);
    const nextHovered = hit ? hit.object.userData.vertex : -1;
    if (nextHovered !== pointerState.hovered) {
      pointerState.hovered = nextHovered;
      renderer.domElement.style.cursor = hit ? "grab" : "default";
      rebuildGeometry();
    }
  });

  function endPointer(event) {
    if (pointerState.mode === "vertex") {
      if (pendingDragTarget !== null) {
        model.dragVertex(pointerState.vertex, pendingDragTarget);
        pendingDragTarget = null;
      }
      model.project(8);
      element("editState").textContent = "Constraints restored";
      setTimeout(() => {
        if (element("editState").textContent === "Constraints restored") {
          element("editState").textContent = "Ready";
        }
      }, 1300);
    }
    pointerState.mode = null;
    pointerState.vertex = -1;
    renderer.domElement.style.cursor = "default";
    if (renderer.domElement.hasPointerCapture(event.pointerId)) {
      renderer.domElement.releasePointerCapture(event.pointerId);
    }
    rebuildGeometry();
  }
  renderer.domElement.addEventListener("pointerup", endPointer);
  renderer.domElement.addEventListener("pointercancel", endPointer);
  renderer.domElement.addEventListener("contextmenu", (event) => event.preventDefault());
  renderer.domElement.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      cameraState.distance = Math.max(
        3,
        Math.min(40, cameraState.distance * Math.exp(event.deltaY * 0.0012)),
      );
      updateCamera();
    },
    { passive: false },
  );

  controls.sample.addEventListener("change", () => {
    const key = controls.sample.value;
    if (samples[key]) loadData(samples[key], key, key);
  });
  controls.open.addEventListener("click", () => controls.file.click());
  controls.file.addEventListener("change", () => {
    const [file] = controls.file.files;
    if (file) readFile(file);
    controls.file.value = "";
  });

  function readFile(file) {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      try {
        loadData(JSON.parse(reader.result), file.name);
      } catch (error) {
        toast(`Could not read ${file.name}: ${error.message}`);
      }
    });
    reader.readAsText(file);
  }

  let dragDepth = 0;
  window.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    element("dropZone").classList.add("visible");
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth -= 1;
    if (dragDepth <= 0) element("dropZone").classList.remove("visible");
  });
  window.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    element("dropZone").classList.remove("visible");
    const [file] = event.dataTransfer.files;
    if (file) readFile(file);
  });

  controls.export.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(model.toJSON(), null, 2)], {
      type: "application/json",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${currentFileName}_edited.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
    toast("The current constrained configuration was exported.");
  });

  controls.play.addEventListener("click", () => setRunning(!running));
  controls.reverse.addEventListener("click", () => {
    direction *= -1;
    controls.reverse.textContent = direction < 0 ? "Forward" : "Reverse";
    updateFlowControls();
  });
  controls.step.addEventListener("click", () => {
    setRunning(false);
    const requested = direction * Number(controls.flowRate.value) * 0.0012;
    model.advanceFlow(
      controls.flowKind.value,
      Number(controls.flowOrder.value),
      requested,
    );
    model.project(3);
    rebuildGeometry();
  });
  controls.flowKind.addEventListener("change", updateFlowControls);
  controls.flowOrder.addEventListener("input", updateFlowControls);
  controls.flowRate.addEventListener("input", updateFlowControls);

  controls.torsion.addEventListener("input", () => {
    setRunning(false);
    pendingTorsion = Number(controls.torsion.value);
    controls.torsionValue.textContent = `${pendingTorsion.toFixed(4)} rad`;
    element("editState").textContent = "Solving…";
  });
  controls.resetShape.addEventListener("click", () => {
    setRunning(false);
    model.restore();
    controls.torsion.value = String(model.torsionAngle);
    rebuildGeometry();
    toast("The initial configuration was restored.");
  });
  controls.setInitial.addEventListener("click", () => {
    model.initialState = model.snapshot();
    model.flowTime = 0;
    updateDiagnostics();
    toast("The current shape is now the initial configuration.");
  });

  controls.ribbonWidth.addEventListener("input", () => {
    controls.widthValue.textContent = Number(controls.ribbonWidth.value).toFixed(2);
    rebuildGeometry();
  });
  controls.opacity.addEventListener("input", () => {
    controls.opacityValue.textContent = Number(controls.opacity.value).toFixed(2);
    surfaceMaterial.opacity = Number(controls.opacity.value);
  });
  [controls.centreline, controls.hinges, controls.handles].forEach((control) =>
    control.addEventListener("change", rebuildGeometry),
  );
  controls.resetView.addEventListener("click", resetCamera);
  controls.capture.addEventListener("click", () => {
    renderer.render(scene, camera);
    const anchor = document.createElement("a");
    anchor.download = `${currentFileName}.png`;
    anchor.href = renderer.domElement.toDataURL("image/png");
    anchor.click();
  });

  const resizeObserver = new ResizeObserver(() => {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
  });
  resizeObserver.observe(viewport);

  function animate(now) {
    requestAnimationFrame(animate);
    const elapsed = Math.min(0.05, Math.max(0, (now - lastFrameTime) / 1000));
    lastFrameTime = now;

    if (pendingTorsion !== null) {
      const requested = pendingTorsion;
      pendingTorsion = null;
      const succeeded = model.setTorsion(requested);
      controls.torsion.value = String(model.torsionAngle);
      element("editState").textContent = succeeded ? "Constraints restored" : "Limit reached";
      if (!succeeded) toast("That torsion value is not reachable on this local branch.");
      rebuildGeometry();
    }
    if (pendingDragTarget !== null && pointerState.vertex >= 0) {
      const target = pendingDragTarget;
      pendingDragTarget = null;
      model.dragVertex(pointerState.vertex, target);
      rebuildGeometry();
    }
    if (running && elapsed > 0) {
      const requested =
        direction * Number(controls.flowRate.value) * elapsed * 0.055;
      model.advanceFlow(
        controls.flowKind.value,
        Number(controls.flowOrder.value),
        requested,
      );
      projectionCounter += 1;
      if (projectionCounter % 12 === 0) model.project(2);
      rebuildGeometry();
    }
    renderer.render(scene, camera);
  }

  async function start() {
    try {
      await loadSampleData();
      populateSamples();
      const initialKey = controls.sample.value;
      loadData(samples[initialKey], initialKey, initialKey);
      requestAnimationFrame(animate);
    } catch (error) {
      const fatal = element("fatalError");
      fatal.textContent = error instanceof Error ? error.message : String(error);
      fatal.classList.add("visible");
      console.error(error);
    }
  }

  start();
})();
