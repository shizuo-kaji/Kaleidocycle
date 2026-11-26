(function initialiseStudio() {
  "use strict";

  const THREE = window.THREE;
  const Mathematics = window.Kaleidocycle;
  const Families = window.KaleidocycleFamilies;
  const Print = window.KaleidocyclePrint;
  const staticFallback = window.KALEIDOCYCLE_SAMPLE_FALLBACK;
  const webDirectory = new URL(".", window.location.href);
  const servedFromWebDirectory = /\/web\/$/.test(webDirectory.pathname);
  const dataBaseUrl = new URL(
    servedFromWebDirectory ? "../data/kaleidocycles/" : "./data/kaleidocycles/",
    webDirectory,
  );
  const element = (id) => document.getElementById(id);
  const viewport = element("viewport");

  if (!THREE || !Mathematics || !Families) {
    const fatal = element("fatalError");
    fatal.textContent =
      "The local application files are incomplete. Check sim.js, families.js and web/vendor/.";
    fatal.classList.add("visible");
    return;
  }

  const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const GENERATED_KEY = "__generated__";
  const SAMPLE_GROUPS = [
    ["critical", "Torsion-critical circles"],
    ["mobius", "Möbius · maximal torsion"],
    ["knot", "Knots and linking"],
    ["explicit", "Explicit theta solutions"],
    ["generic", "Generic · non-critical"],
    ["counterexample", "Counterexamples"],
  ];

  let samples = {};
  let catalogue = { default: null, samples: [] };
  let sampleSource = "data/";

  const controls = {
    sample: element("sampleSelect"),
    open: element("openFile"),
    export: element("exportFile"),
    file: element("fileInput"),
    description: element("modelDescription"),
    chips: element("modelChips"),
    labCriticalTab: element("labCriticalTab"),
    labKnotTab: element("labKnotTab"),
    labCritical: element("labCritical"),
    labKnot: element("labKnot"),
    labN: element("labN"),
    labNValue: element("labNValue"),
    labWinding: element("labWinding"),
    labLevel: element("labLevel"),
    labMirror: element("labMirror"),
    labCells: element("labCells"),
    labTurns: element("labTurns"),
    labKnotMirror: element("labKnotMirror"),
    labInfo: element("labInfo"),
    labGenerate: element("labGenerate"),
    labPrincipal: element("labPrincipal"),
    phase: element("phase"),
    phaseValue: element("phaseValue"),
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
    mirror: element("mirrorShape"),
    analysisButtons: {
      profile: element("analysisProfile"),
      portrait: element("analysisPortrait"),
      closing: element("analysisClosing"),
    },
    analysisCanvas: element("analysisCanvas"),
    analysisHint: element("analysisHint"),
    analysisTag: element("analysisTag"),
    colourMode: element("colourMode"),
    ribbonWidth: element("ribbonWidth"),
    widthValue: element("widthValue"),
    opacity: element("surfaceOpacity"),
    opacityValue: element("opacityValue"),
    centreline: element("showCentreline"),
    hinges: element("showHinges"),
    handles: element("showHandles"),
    stabilise: element("stabilise"),
    resetView: element("resetView"),
    autoRotate: element("autoRotate"),
    capture: element("captureView"),
    record: element("recordView"),
    copyLink: element("copyLink"),
    help: element("toggleHelp"),
    helpOverlay: element("helpOverlay"),
    printCard: element("printCard"),
    printTag: element("printTag"),
    printShow: element("printShow"),
    printWidth: element("printWidth"),
    printEdge: element("printEdge"),
    printThickness: element("printThickness"),
    printHinge: element("printHinge"),
    printRing: element("printRing"),
    printPinFit: element("printPinFit"),
    printPinTurn: element("printPinTurn"),
    printKnuckleGap: element("printKnuckleGap"),
    printEdgeGap: element("printEdgeGap"),
    printInfo: element("printInfo"),
    printPlate: element("printPlate"),
    printLink: element("printLink"),
    printAssembly: element("printAssembly"),
    printCheck: element("printCheck"),
    printClosest: element("printClosest"),
    printStatus: element("printStatus"),
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
  // The vendored three.js (r148) defaults to legacy light units, which wash the
  // vertex colours out to white with these intensities.
  if ("physicallyCorrectLights" in renderer) renderer.physicallyCorrectLights = true;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
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

  // Printable twisted-plate links (one instanced mesh per part).
  const printGroup = new THREE.Group();
  printGroup.visible = false;
  modelGroup.add(printGroup);
  const printMaterials = {
    plate: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.03 }),
    ring: new THREE.MeshStandardMaterial({ color: 0xe8efe4, roughness: 0.5, metalness: 0.04 }),
    pin: new THREE.MeshStandardMaterial({ color: 0xd9c38f, roughness: 0.38, metalness: 0.32 }),
  };
  const printContactColour = new THREE.Color(0xff5a4f).convertSRGBToLinear();
  const printState = {
    enabled: false,
    loading: null,
    builder: null,
    unit: null,
    instances: {},
    timer: null,
    building: null,
    error: "",
    contacts: new Set(),
    closestState: null,
    checking: false,
    modelRef: null,
  };
  let lastDisplay = null;

  let model;
  let modelMetadata = {};
  let currentSampleKey = "";
  let running = false;
  let direction = 1;
  let lastFrameTime = performance.now();
  let projectionCounter = 0;
  let pendingTorsion = null;
  let pendingDragTarget = null;
  let pendingPhase = null;
  let toastTimer;
  let currentFileName = "kaleidocycle";
  let autoRotate = false;
  let analysisMode = "profile";
  let labMode = "critical";
  let recorder = null;

  // Exact family behind the current model, if any (drives the phase slider).
  let currentCircle = null;
  let currentPhase = 0;
  let generatedSource = null;

  // Display-only rotation that removes rigid drift (raw → displayed).
  let displayRotation = IDENTITY.slice();
  let displayReference = null;
  let displayedPoints = null;

  const closingCurveCache = new Map();

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

  const rotate = (matrix, v) => [
    matrix[0] * v[0] + matrix[1] * v[1] + matrix[2] * v[2],
    matrix[3] * v[0] + matrix[4] * v[1] + matrix[5] * v[2],
    matrix[6] * v[0] + matrix[7] * v[1] + matrix[8] * v[2],
  ];
  const rotateBack = (matrix, v) => [
    matrix[0] * v[0] + matrix[3] * v[1] + matrix[6] * v[2],
    matrix[1] * v[0] + matrix[4] * v[1] + matrix[7] * v[2],
    matrix[2] * v[0] + matrix[5] * v[1] + matrix[8] * v[2],
  ];

  function toast(message) {
    const target = element("toast");
    target.textContent = message;
    target.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => target.classList.remove("visible"), 2600);
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

  function isOriented(metadata) {
    const value = metadata?.oriented;
    return value === true || String(value).toLowerCase() === "true";
  }

  function friendlyName(key, data) {
    if (data.metadata?.label) return data.metadata.label;
    const n = data.n || data.metadata?.n || "?";
    const family = key.startsWith("generic") ? "Generic" : "Möbius";
    const orientation = isOriented(data.metadata) ? "oriented" : "anti-oriented";
    const locus = key.includes("noncritical") ? " · non-critical" : "";
    return `${family} K${n} · ${orientation}${locus}`;
  }

  function populateSamples() {
    controls.sample.replaceChildren();
    const groups = new Map(SAMPLE_GROUPS.map(([kind, label]) => {
      const group = document.createElement("optgroup");
      group.label = label;
      return [kind, group];
    }));
    const other = document.createElement("optgroup");
    other.label = "Other";
    [...catalogue.samples]
      .sort((left, right) => left.n - right.n || left.name.localeCompare(right.name))
      .forEach((entry) => {
        const data = samples[entry.name];
        if (!data) return;
        const option = document.createElement("option");
        option.value = entry.name;
        option.textContent = friendlyName(entry.name, data);
        (groups.get(entry.kind) || other).append(option);
      });
    [...groups.values(), other].forEach((group) => {
      if (group.children.length) controls.sample.append(group);
    });
    controls.sample.value = samples[catalogue.default]
      ? catalogue.default
      : catalogue.samples[0]?.name;
  }

  function setGeneratedOption(label) {
    let group = controls.sample.querySelector("optgroup[data-generated]");
    if (!label) {
      group?.remove();
      return;
    }
    if (!group) {
      group = document.createElement("optgroup");
      group.label = "Model lab";
      group.dataset.generated = "true";
      controls.sample.prepend(group);
    }
    const option = document.createElement("option");
    option.value = GENERATED_KEY;
    option.textContent = label;
    group.replaceChildren(option);
  }

  function threeVector(values) {
    return new THREE.Vector3(values[0], values[1], values[2]);
  }

  function replaceGeometry(object, geometry) {
    object.geometry.dispose();
    object.geometry = geometry;
  }

  const paletteStart = new THREE.Color(0x84d6b0);
  const paletteMiddle = new THREE.Color(0xc9f28c);
  const paletteEnd = new THREE.Color(0x73a9de);
  const curvatureNegative = new THREE.Color(0x3f7fd6);
  const curvatureZero = new THREE.Color(0xa9b8ad);
  const curvaturePositive = new THREE.Color(0xeb8f3c);
  const alternatingA = new THREE.Color(0xc9f28c);
  const alternatingB = new THREE.Color(0x5b8fc9);
  const monochrome = new THREE.Color(0xb9d8c4);

  function gradient(index, count) {
    const position = count <= 1 ? 0 : index / (count - 1);
    return position < 0.5
      ? paletteStart.clone().lerp(paletteMiddle, position * 2)
      : paletteMiddle.clone().lerp(paletteEnd, (position - 0.5) * 2);
  }

  function hingeCurvature(index) {
    return index < model.n ? model.curvatures[index] : model.sign * model.curvatures[0];
  }

  function linkColours(index, curvatureScale) {
    const mode = controls.colourMode.value;
    if (mode === "curvature") {
      const colour = (value) => {
        const t = Math.tanh(value / curvatureScale);
        return t < 0
          ? curvatureZero.clone().lerp(curvatureNegative, -t)
          : curvatureZero.clone().lerp(curvaturePositive, t);
      };
      const first = colour(hingeCurvature(index));
      const second = colour(hingeCurvature(index + 1));
      return [first, first, second, second];
    }
    let colour;
    if (mode === "alternating") colour = index % 2 === 0 ? alternatingA : alternatingB;
    else if (mode === "mono") colour = monochrome;
    else colour = gradient(index, model.n);
    return [colour, colour, colour, colour];
  }

  function displayGeometry() {
    const raw = model.displayVertices();
    const rawBinormals = model.configuration.frames.map((frame) => [
      frame[2], frame[5], frame[8],
    ]);
    if (!controls.stabilise.checked) {
      displayRotation = IDENTITY.slice();
      displayReference = null;
    } else if (
      displayReference &&
      displayReference.length === raw.length &&
      pointerState.mode !== "vertex"
    ) {
      displayRotation = Mathematics.bestRotation(
        raw.slice(0, model.n),
        displayReference.slice(0, model.n),
      );
    }
    const points = raw.map((point) => rotate(displayRotation, point));
    const binormals = rawBinormals.map((vector) => rotate(displayRotation, vector));
    if (controls.stabilise.checked) displayReference = points;
    return { points, binormals };
  }

  function rebuildGeometry() {
    if (!model) return;
    const positions = [];
    const colours = [];
    const edgePositions = [];
    const hingePositions = [];
    const display = displayGeometry();
    displayedPoints = display.points;
    lastDisplay = display;
    const points = display.points.map(threeVector);
    const binormals = display.binormals.map(threeVector);
    const width = Number(controls.ribbonWidth.value);
    // Root-mean-square curvature: typical values saturate the diverging map.
    const curvatureScale = Math.max(
      0.1,
      Math.sqrt(
        model.curvatures.reduce((sum, value) => sum + value * value, 0) / model.n,
      ),
    );
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
      // Vertex colours are linear; the palette is specified in sRGB.
      const vertexColours = linkColours(index, curvatureScale).map((colour) =>
        colour.clone().convertSRGBToLinear());
      triangleIndices.forEach((triangle) => {
        triangle.forEach((vertexIndex) => {
          positions.push(...vertices[vertexIndex].toArray());
          const colour = vertexColours[vertexIndex];
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
    const showPrint = printState.enabled && Boolean(printState.unit);
    mesh.visible = !showPrint;
    edges.visible = !showPrint;
    printGroup.visible = showPrint;
    if (printState.enabled) updatePrintInstances(display, curvatureScale);
    updateDiagnostics();
  }

  // ---------------------------------------------------------------------------
  // 3D print: twisted-plate links built in the browser with manifold-3d.

  const PRINT_ENGINE_URL = "vendor/manifold-classic.js";
  const printMatrix = new THREE.Matrix4();
  const nextTask = () => new Promise((resolve) => setTimeout(resolve, 0));

  function printNumber(input, fallback) {
    if (input.value === "") return fallback;
    const value = Number(input.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function printOverrides() {
    const defaults = Print.DEFAULTS;
    return {
      ribbon_width: printNumber(controls.printWidth, defaults.ribbon_width),
      edge_length: printNumber(controls.printEdge, defaults.edge_length),
      body_thickness: printNumber(controls.printThickness, defaults.body_thickness),
      hinge_type: controls.printHinge.value,
      with_ring: controls.printRing.checked,
      pin_fit_clearance: printNumber(controls.printPinFit, defaults.pin_fit_clearance),
      pin_clearance: printNumber(controls.printPinTurn, defaults.pin_clearance),
      hinge_gap: printNumber(controls.printKnuckleGap, null),
      gap: printNumber(controls.printEdgeGap, defaults.gap),
    };
  }

  function setPrintTag(text) {
    controls.printTag.textContent = text;
  }

  function setPrintStatus(text, isError = false) {
    controls.printStatus.textContent = text;
    controls.printStatus.classList.toggle("error", isError);
  }

  function loadPrintEngine() {
    if (printState.builder) return Promise.resolve(printState.builder);
    if (printState.loading) return printState.loading;
    setPrintTag("Loading…");
    printState.loading = new Promise((resolve, reject) => {
      if (window.KaleidocycleManifold) {
        resolve();
        return;
      }
      const script = document.createElement("script");
      script.src = PRINT_ENGINE_URL;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Could not load web/${PRINT_ENGINE_URL}.`));
      document.head.append(script);
    })
      .then(() => window.KaleidocycleManifold.load())
      .then((wasm) => {
        printState.builder = Print.createBuilder(wasm);
        return printState.builder;
      });
    printState.loading.catch(() => {
      printState.loading = null;
    });
    return printState.loading;
  }

  function currentPlacements(display = lastDisplay) {
    return Print.linkPlacements(display.points, display.binormals);
  }

  function printKey(overrides, alpha) {
    return JSON.stringify({ ...overrides, alpha: Math.round(alpha * 1e5) });
  }

  function disposePrintUnit() {
    const unit = printState.unit;
    if (!unit) return;
    Object.values(unit.manifolds).forEach((item) => item?.delete());
    printState.unit = null;
  }

  function geometryFromMesh({ positions, indices }) {
    const indexed = new THREE.BufferGeometry();
    indexed.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    indexed.setIndex(new THREE.BufferAttribute(indices, 1));
    const flat = indexed.toNonIndexed();
    indexed.dispose();
    flat.computeVertexNormals();
    return flat;
  }

  function createPrintMeshes() {
    Object.values(printState.instances).forEach((instanced) => {
      printGroup.remove(instanced);
      instanced.geometry.dispose();
      instanced.dispose?.();
    });
    printState.instances = {};
    const unit = printState.unit;
    if (!unit || !model) return;
    Object.entries(unit.meshes).forEach(([part, arrays]) => {
      const instanced = new THREE.InstancedMesh(
        geometryFromMesh(arrays),
        printMaterials[part],
        model.n,
      );
      instanced.castShadow = true;
      instanced.receiveShadow = true;
      instanced.frustumCulled = false;
      printState.instances[part] = instanced;
      printGroup.add(instanced);
    });
  }

  function updatePrintInstances(display, curvatureScale) {
    const unit = printState.unit;
    if (!unit || !display) return;
    if (printState.modelRef !== model) {
      printState.modelRef = model;
      printState.contacts.clear();
      printState.closestState = null;
      controls.printClosest.disabled = true;
    }
    if (printState.instances.plate?.count !== model.n) createPrintMeshes();
    const placements = currentPlacements(display);
    const scale = 1 / unit.cfg.edge_length;
    const zb = unit.cfg.zBottomHinge;
    const { plate, ring, pin } = printState.instances;
    placements.forEach((placement, index) => {
      printMatrix.fromArray(Print.placementMatrix(placement, scale, zb));
      plate.setMatrixAt(index, printMatrix);
      ring?.setMatrixAt(index, printMatrix);
      if (pin) {
        printMatrix.fromArray(Print.placementMatrix(placement, scale, zb, false));
        pin.setMatrixAt(index, printMatrix);
      }
      const colour = printState.contacts.has(index)
        ? printContactColour
        : linkColours(index, curvatureScale)[0].clone().convertSRGBToLinear();
      plate.setColorAt(index, colour);
    });
    [plate, ring, pin].forEach((instanced) => {
      if (instanced) instanced.instanceMatrix.needsUpdate = true;
    });
    if (plate.instanceColor) plate.instanceColor.needsUpdate = true;
    // The torsion moved (slider, Model lab): rebuild the link.
    if (Math.abs(placements[0].alpha - unit.alpha) > 2e-4) schedulePrintBuild();
  }

  function schedulePrintBuild(delay = 160) {
    clearTimeout(printState.timer);
    printState.timer = setTimeout(() => {
      printState.timer = null;
      buildPrintUnit().catch((error) => console.error(error));
    }, delay);
  }

  async function buildPrintUnit() {
    if (!model || !lastDisplay) return null;
    if (printState.building) return printState.building;
    printState.building = (async () => {
      let builder;
      try {
        builder = await loadPrintEngine();
      } catch (error) {
        printState.error = error.message;
        setPrintTag("Error");
        setPrintStatus(error.message, true);
        return null;
      }
      const overrides = printOverrides();
      const alpha = currentPlacements()[0].alpha;
      let cfg;
      let parts;
      let pin = null;
      const start = performance.now();
      try {
        cfg = Print.validate(Print.config({ ...overrides, torsion_angle: alpha }));
        parts = builder.unit(cfg);
        if (cfg.usesPins) pin = builder.pin(cfg);
      } catch (error) {
        printState.error = error instanceof Error ? error.message : String(error);
        setPrintTag("Check settings");
        setPrintStatus(`Cannot build the link: ${printState.error}.`, true);
        return null;
      }
      const meshes = { plate: Print.meshArrays(parts.plate) };
      if (parts.ring) meshes.ring = Print.meshArrays(parts.ring);
      if (pin) meshes.pin = Print.meshArrays(pin);
      disposePrintUnit();
      printState.unit = {
        cfg,
        alpha,
        key: printKey(overrides, alpha),
        meshes,
        manifolds: { plate: parts.plate, ring: parts.ring || null, pin },
        triangles: Object.values(meshes).reduce((sum, item) => sum + item.indices.length / 3, 0),
        millis: performance.now() - start,
      };
      printState.error = "";
      printState.contacts.clear();
      printState.closestState = null;
      controls.printClosest.disabled = true;
      createPrintMeshes();
      updatePrintInfo();
      setPrintTag(printState.enabled ? "Live" : "Ready");
      setPrintStatus(
        "Links follow the flow, the phase slider and editing; a torsion change rebuilds them. " +
          "Check contacts traces one cycle of the selected flow.",
      );
      rebuildGeometry();
      return printState.unit;
    })();
    try {
      return await printState.building;
    } finally {
      printState.building = null;
    }
  }

  /** The link for the current settings and torsion, built if needed. */
  async function ensurePrintUnit() {
    if (!model) return null;
    if (!lastDisplay) rebuildGeometry();
    const key = printKey(printOverrides(), currentPlacements()[0].alpha);
    if (printState.unit?.key === key) return printState.unit;
    clearTimeout(printState.timer);
    printState.timer = null;
    return buildPrintUnit();
  }

  function updatePrintInfo() {
    const unit = printState.unit;
    if (!unit) return;
    const cfg = unit.cfg;
    const mm = (value) => `${Number(value.toFixed(2))}`;
    const hinge = cfg.usesPins
      ? `pin bores ${mm(2 * cfg.fixedBoreRadius)} / ${mm(2 * cfg.freeBoreRadius)} mm`
      : `${mm(cfg.cone_diameter)} mm snap-in cones`;
    controls.printInfo.classList.remove("error");
    controls.printInfo.textContent =
      `μ = ${cfg.torsionDegrees.toFixed(2)}° · link ${mm(cfg.ribbon_width)} × ${mm(cfg.edge_length)} mm, ` +
      `${mm(cfg.body_thickness)} mm plate · ${hinge}, knuckle gap ${mm(cfg.hingeGap)} mm · ` +
      `${unit.triangles.toLocaleString("en-GB")} triangles, ${Math.round(unit.millis)} ms`;
  }

  function setPrintEnabled(enabled) {
    printState.enabled = enabled;
    controls.printShow.checked = enabled;
    if (enabled) {
      setPrintTag(printState.unit ? "Live" : "Loading…");
      ensurePrintUnit().catch((error) => console.error(error));
    } else {
      setPrintTag(printState.unit ? "Ready" : "Off");
    }
    rebuildGeometry();
  }

  function printBaseName() {
    const orientation = model.sign > 0 ? "oriented" : "anti_oriented";
    const name = currentFileName && currentFileName !== "kaleidocycle"
      ? currentFileName
      : `k${model.n}_${orientation}`;
    return name.replace(/[^\w.-]+/g, "_");
  }

  async function withPrintUnit(label, action) {
    const buttons = [controls.printPlate, controls.printLink, controls.printAssembly, controls.printCheck];
    buttons.forEach((button) => { button.disabled = true; });
    try {
      setPrintStatus(`${label}…`);
      const unit = await ensurePrintUnit();
      if (!unit) return;
      await nextTask();
      await action(unit);
    } catch (error) {
      console.error(error);
      setPrintStatus(error instanceof Error ? error.message : String(error), true);
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
    }
  }

  function unitParts(unit) {
    const parts = [{ name: "plate", mesh: unit.meshes.plate }];
    if (unit.meshes.ring) parts.push({ name: "ring", mesh: unit.meshes.ring });
    return parts;
  }

  function exportPrintPlate() {
    return withPrintUnit("Writing the print plate", async (unit) => {
      const parts = unitParts(unit);
      const objects = [{ key: "unit", name: "twisted_plate", parts }];
      const { lo, hi } = Print.bounds(parts.map((part) => part.mesh));
      let pinLength = 0;
      let pinWidth = 0;
      if (unit.meshes.pin) {
        const pinBox = Print.bounds([unit.meshes.pin]);
        pinLength = pinBox.hi[0] - pinBox.lo[0];
        pinWidth = pinBox.hi[1] - pinBox.lo[1];
        objects.push({ key: "pin", name: "hinge_pin", parts: [{ name: "pin", mesh: unit.meshes.pin }] });
      }
      const layout = Print.plateLayout(model.n, hi[0] - lo[0], hi[1] - lo[1], {
        pinCount: unit.meshes.pin ? model.n : 0,
        pinLength,
        pinWidth,
      });
      const items = [
        ...layout.units.map(([x, y]) => ({ key: "unit", x, y })),
        ...layout.pins.map(([x, y]) => ({ key: "pin", x, y })),
      ];
      const name = `${printBaseName()}_plate`;
      const bytes = await Print.threeMF({ title: name, objects, items });
      download(new Blob([bytes], { type: "model/3mf" }), `${name}.3mf`);
      const halfWidth = 0.5 * Math.max(hi[0] - lo[0], pinLength);
      const halfDepth = 0.5 * Math.max(hi[1] - lo[1], pinWidth);
      const xs = items.map((item) => item.x);
      const ys = items.map((item) => item.y);
      const span = Math.max(
        Math.max(...xs) + halfWidth - (Math.min(...xs) - halfWidth),
        Math.max(...ys) + halfDepth - (Math.min(...ys) - halfDepth),
      );
      setPrintStatus(
        `Saved ${name}.3mf: ${model.n} links${unit.meshes.pin ? ` and ${model.n} pins` : ""} ` +
          `around the centre of a 256 mm bed (${Math.ceil(span)} mm across).` +
          (span > 256 ? " The layout is larger than a 256 mm bed: print it in batches." : ""),
      );
    });
  }

  function exportPrintLink() {
    return withPrintUnit("Writing the link", async (unit) => {
      const name = `${printBaseName()}_unit`;
      const buffer = Print.stl(unitParts(unit).map((part) => part.mesh), name);
      download(new Blob([buffer], { type: "model/stl" }), `${name}.stl`);
      setPrintStatus(`Saved ${name}.stl (one link, in its printing orientation).`);
    });
  }

  function exportPrintAssembly() {
    return withPrintUnit("Writing the assembly", async (unit) => {
      const E = unit.cfg.edge_length;
      const zb = unit.cfg.zBottomHinge;
      const placements = Print.linkPlacements(
        lastDisplay.points.map((point) => point.map((value) => value * E)),
        lastDisplay.binormals,
      );
      const meshes = [];
      placements.forEach((placement) => {
        const matrix = Print.placementMatrix(placement, 1, zb);
        unitParts(unit).forEach((part) => meshes.push({ ...part.mesh, matrix }));
        if (unit.meshes.pin) {
          meshes.push({ ...unit.meshes.pin, matrix: Print.placementMatrix(placement, 1, zb, false) });
        }
      });
      const name = `${printBaseName()}_assembly`;
      download(new Blob([Print.stl(meshes, name)], { type: "model/stl" }), `${name}.stl`);
      setPrintStatus(`Saved ${name}.stl (the assembled ring in the current pose).`);
    });
  }

  function checkPrintContacts() {
    return withPrintUnit("Tracing one cycle of the flow", async (unit) => {
      setRunning(false);
      const kind = controls.flowKind.value;
      const order = Number(controls.flowOrder.value);
      const copy = new Mathematics.Model({ n: model.n, hinges: model.hinges() });
      const trace = Print.traceCycle(copy, kind, order);
      const E = unit.cfg.edge_length;
      const zb = unit.cfg.zBottomHinge;
      const placementsOf = (state) => Print.linkPlacements(
        state.points.map((point) => point.map((v) => v * E)),
        state.binormals,
      );
      const angleSets = trace.states.map((state) => Print.foldAngles(placementsOf(state), unit.alpha));
      const stride = Math.max(1, Math.ceil(trace.states.length / 72));
      const sampledIndices = trace.states.map((_, index) => index).filter((index) => index % stride === 0);
      const poses = sampledIndices.map((index) =>
        placementsOf(trace.states[index]).map((placement) => Print.placementMatrix(placement, 1, zb)));
      const bodies = printState.builder.linkBodies(unit.manifolds, unit.manifolds.pin, unit.cfg);
      let result;
      try {
        result = await Print.linkGaps(bodies, unit.cfg, {
          poses,
          angles: angleSets.flat(),
          onProgress: async (done, total) => {
            setPrintStatus(`Measuring the gaps between links: ${Math.round((100 * done) / total)} %…`);
            await nextTask();
          },
        });
      } finally {
        bodies.delete();
      }
      const limit = Print.MIN_CLEARANCE;
      const n = model.n;
      // Links in contact: non-adjacent pairs, and hinges that reach a contact angle.
      const contacts = new Set(result.nonAdjacentContacts.flatMap((pair) => [pair.i, pair.j]));
      const near = (theta) => result.adjacentContactAngles.some(
        (angle) => Math.abs(angle - theta) <= result.fineStep,
      );
      const hingesInContact = new Set();
      angleSets.forEach((angles) => angles.forEach((theta, hinge) => {
        if (near(theta)) hingesInContact.add(hinge);
      }));
      hingesInContact.forEach((hinge) => {
        contacts.add(hinge);
        contacts.add((hinge + 1) % n);
      });
      printState.contacts = contacts;
      // Pose of the closest approach.
      const adjacentFirst = result.adjacent &&
        (!result.nonAdjacent || result.adjacent.gap <= result.nonAdjacent.gap);
      let closestIndex = null;
      if (adjacentFirst) {
        let bestDistance = Infinity;
        angleSets.forEach((angles, index) => angles.forEach((theta) => {
          const distance = Math.abs(theta - result.adjacent.theta);
          if (distance < bestDistance) {
            bestDistance = distance;
            closestIndex = index;
          }
        }));
      } else if (result.nonAdjacent) {
        closestIndex = sampledIndices[result.nonAdjacent.pose];
      }
      printState.closestState = closestIndex === null ? null : trace.states[closestIndex].state;
      controls.printClosest.disabled = !printState.closestState;
      const flow = kind === "sine-gordon" ? "the sine–Gordon flow" : `X${order}`;
      const span = trace.periodic
        ? `one cycle of ${flow}`
        : `${trace.states.length} steps of ${flow}, which did not return to the start`;
      const describe = (item, links) => {
        if (!item) return "—";
        if (item.gap >= 3) return "more than 3 mm";
        return `${item.gap.toFixed(2)} mm${links ? ` (links ${item.i + 1} and ${item.j + 1})` : ""}`;
      };
      const summary =
        `adjacent links away from their hinge ${describe(result.adjacent, false)}, ` +
        `other links ${describe(result.nonAdjacent, true)}`;
      const adjacentHit = result.adjacentContactAngles.length > 0;
      const otherHits = result.nonAdjacentContacts.length;
      if (adjacentHit || otherHits) {
        const counts = [
          adjacentHit ? "adjacent links" : "",
          otherHits ? `${otherHits} pair(s) of links that do not share a hinge` : "",
        ].filter(Boolean).join(" and ");
        setPrintStatus(
          `Contact over ${span}: ${counts} come closer than ${limit} mm (shown in red). ` +
            `Closest: ${summary}. Narrow the ribbon or thin the plates.`,
          true,
        );
      } else {
        setPrintStatus(
          `OK over ${span}: ${summary} (${limit} mm required). ` +
            "The hinge fit itself is checked in Python.",
        );
      }
      rebuildGeometry();
    });
  }

  function showClosestPose() {
    if (!printState.closestState) return;
    setRunning(false);
    model.restore(printState.closestState);
    currentCircle = null;
    updatePhaseControl();
    rebuildGeometry();
    toast("Moved to the pose of the closest approach.");
  }

  function metric(value) {
    if (!Number.isFinite(value)) return "—";
    return Math.abs(value) < 1e-3 ? value.toExponential(2) : value.toFixed(6);
  }

  let lastDiagnostics = null;

  function updateDiagnostics() {
    if (!model) return;
    const diagnostics = model.diagnostics();
    lastDiagnostics = diagnostics;
    element("metricClosure").textContent = metric(diagnostics.closure);
    element("metricMonodromy").textContent = metric(diagnostics.monodromy);
    element("metricE1").textContent = metric(diagnostics.hamiltonian1);
    element("metricE2").textContent = metric(diagnostics.hamiltonian2);
    element("metricWr").textContent = metric(diagnostics.writhe);
    element("metricLk").textContent = metric(diagnostics.linking);
    element("metricTorsion").textContent = (model.torsionAngle / Math.PI).toFixed(6);
    element("metricCritical").textContent = Number.isFinite(diagnostics.criticality)
      ? diagnostics.criticality.toExponential(2)
      : "—";
    element("closureStatus").textContent = diagnostics.closure.toExponential(1);
    const closurePill = element("closurePill");
    closurePill.classList.toggle("warn", diagnostics.closure >= 1e-6);
    closurePill.classList.toggle("bad", diagnostics.closure >= 1e-3);
    element("flowTime").textContent = `t = ${model.flowTime.toExponential(2)}`;
    controls.torsionValue.textContent = `${model.torsionAngle.toFixed(4)} rad`;
    const critical = diagnostics.criticality < 1e-6;
    controls.analysisTag.textContent = critical ? "Critical" : "Not critical";
    drawAnalysis();
  }

  // ---------------------------------------------------------------------
  // Analysis plots
  // ---------------------------------------------------------------------

  const plotColours = {
    axis: "rgba(233, 245, 235, 0.22)",
    grid: "rgba(233, 245, 235, 0.07)",
    text: "#98a59b",
    ink: "#eff5ee",
    lime: "#c9f28c",
    blue: "#75bff2",
    amber: "#f1bc6a",
    faint: "rgba(239, 245, 238, 0.28)",
  };

  function prepareCanvas() {
    const canvas = controls.analysisCanvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(10, canvas.clientWidth);
    const height = Math.max(10, canvas.clientHeight);
    if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
    if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    return { context, width, height };
  }

  function drawProfile() {
    const { context, width, height } = prepareCanvas();
    const kappa = model.curvatures;
    const initial = model.initialState.curvatures;
    const maximum = Math.max(0.5, ...kappa.map(Math.abs), ...initial.map(Math.abs)) * 1.12;
    const left = 30;
    const right = width - 8;
    const top = 10;
    const bottom = height - 18;
    const zero = 0.5 * (top + bottom);
    const scaleY = (value) => zero - (value / maximum) * (0.5 * (bottom - top));
    const slot = (right - left) / model.n;
    context.strokeStyle = plotColours.axis;
    context.beginPath();
    context.moveTo(left, zero);
    context.lineTo(right, zero);
    context.stroke();
    context.fillStyle = plotColours.text;
    context.textAlign = "right";
    context.fillText(maximum.toFixed(1), left - 4, top + 8);
    context.fillText((-maximum).toFixed(1), left - 4, bottom);
    context.fillText("0", left - 4, zero + 3);
    context.textAlign = "left";
    context.fillText("κₙ", left + 2, top + 8);
    context.textAlign = "right";
    context.fillText(`n = 0…${model.n - 1}`, right, height - 4);
    kappa.forEach((value, index) => {
      const x = left + slot * (index + 0.15);
      const w = Math.max(1, slot * 0.7);
      context.fillStyle = value >= 0 ? plotColours.amber : plotColours.blue;
      const y = scaleY(value);
      context.fillRect(x, Math.min(y, zero), w, Math.abs(zero - y));
      context.strokeStyle = plotColours.faint;
      const y0 = scaleY(initial[index]);
      context.beginPath();
      context.moveTo(x - 1, y0);
      context.lineTo(x + w + 1, y0);
      context.stroke();
    });
    controls.analysisHint.textContent =
      "Bars: current Cayley curvatures κₙ = 2 tan(φₙ/2). Ticks: the initial state. " +
      "Hierarchy flows move these as travelling waves while conserving E₁, E₂ and μ.";
  }

  function marchingSquares(context, fn, range, toScreen, resolution = 96) {
    const step = (2 * range) / resolution;
    const values = [];
    for (let row = 0; row <= resolution; row += 1) {
      const line = [];
      for (let col = 0; col <= resolution; col += 1) {
        line.push(fn(-range + col * step, -range + row * step));
      }
      values.push(line);
    }
    const interpolate = (a, b, va, vb) => a + ((b - a) * va) / (va - vb);
    context.beginPath();
    for (let row = 0; row < resolution; row += 1) {
      for (let col = 0; col < resolution; col += 1) {
        const x0 = -range + col * step;
        const y0 = -range + row * step;
        const x1 = x0 + step;
        const y1 = y0 + step;
        const v = [
          values[row][col], values[row][col + 1],
          values[row + 1][col + 1], values[row + 1][col],
        ];
        const crossings = [];
        if ((v[0] > 0) !== (v[1] > 0)) crossings.push([interpolate(x0, x1, v[0], v[1]), y0]);
        if ((v[1] > 0) !== (v[2] > 0)) crossings.push([x1, interpolate(y0, y1, v[1], v[2])]);
        if ((v[2] > 0) !== (v[3] > 0)) crossings.push([interpolate(x1, x0, v[2], v[3]), y1]);
        if ((v[3] > 0) !== (v[0] > 0)) crossings.push([x0, interpolate(y1, y0, v[3], v[0])]);
        for (let index = 0; index + 1 < crossings.length; index += 2) {
          const [ax, ay] = toScreen(...crossings[index]);
          const [bx, by] = toScreen(...crossings[index + 1]);
          context.moveTo(ax, ay);
          context.lineTo(bx, by);
        }
      }
    }
    context.stroke();
  }

  function drawPortrait() {
    const { context, width, height } = prepareCanvas();
    const n = model.n;
    const length = model.sign === -1 ? 2 * n : n;
    const extended = Array.from({ length: length + 1 }, (_, index) => {
      const wraps = Math.floor(index / n);
      const factor = model.sign === -1 && wraps % 2 === 1 ? -1 : 1;
      return factor * model.curvatures[index % n];
    });
    const range = Math.max(0.6, 1.18 * Math.max(...model.curvatures.map(Math.abs)));
    const size = Math.min(width - 20, height - 20);
    const originX = width / 2;
    const originY = height / 2;
    const toScreen = (x, y) => [
      originX + (x / range) * (size / 2),
      originY - (y / range) * (size / 2),
    ];
    context.strokeStyle = plotColours.grid;
    context.strokeRect(originX - size / 2, originY - size / 2, size, size);
    context.strokeStyle = plotColours.axis;
    context.beginPath();
    context.moveTo(originX - size / 2, originY);
    context.lineTo(originX + size / 2, originY);
    context.moveTo(originX, originY - size / 2);
    context.lineTo(originX, originY + size / 2);
    context.stroke();
    context.fillStyle = plotColours.text;
    context.textAlign = "left";
    context.fillText("κₙ₊₁", originX + 4, originY - size / 2 + 10);
    context.textAlign = "right";
    context.fillText("κₙ", originX + size / 2 - 2, originY - 4);
    context.fillText(`±${range.toFixed(2)}`, width - 6, height - 5);

    const diagnostics = lastDiagnostics || model.diagnostics();
    const critical = diagnostics.criticality < 1e-6;
    const recurrence = Mathematics.recurrenceDiagnostics(
      model.curvatures,
      model.torsionAngle,
      model.sign,
    );
    const lambda = recurrence.multiplier;
    const level = recurrence.qrt.reduce((sum, value) => sum + value, 0) / n;
    const invariant = (x, y) =>
      (1 + (x * x) / 4) * (1 + (y * y) / 4) - (lambda / 4) * x * y - level;
    context.strokeStyle = critical ? plotColours.lime : "rgba(201, 242, 140, 0.32)";
    context.lineWidth = critical ? 1.6 : 1;
    if (!critical) context.setLineDash([4, 4]);
    marchingSquares(context, invariant, range, toScreen);
    context.setLineDash([]);
    context.lineWidth = 1;

    context.strokeStyle = plotColours.faint;
    context.beginPath();
    for (let index = 0; index < length; index += 1) {
      const [x, y] = toScreen(extended[index], extended[index + 1]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.closePath();
    context.stroke();
    for (let index = 0; index < length; index += 1) {
      const [x, y] = toScreen(extended[index], extended[index + 1]);
      context.fillStyle = index === 0 ? plotColours.amber : plotColours.ink;
      context.beginPath();
      context.arc(x, y, index === 0 ? 3.4 : 2.4, 0, 2 * Math.PI);
      context.fill();
    }
    controls.analysisHint.textContent = critical
      ? "Critical: every pair (κₙ, κₙ₊₁) lies on the QRT biquadric " +
        "(1 + x²/4)(1 + y²/4) − (λ/4)xy = J with λ = 2cos μ + E₂/N."
      : "Not critical: the pairs (κₙ, κₙ₊₁) scatter off the QRT level curve (dashed). " +
        "Load or generate a critical circle to see them lie on it.";
  }

  function closingCurve(n, winding) {
    const key = `${n}:${winding}`;
    if (!closingCurveCache.has(key)) {
      closingCurveCache.set(key, Families.closingAngleCurve(n, winding, 160));
    }
    return closingCurveCache.get(key);
  }

  function drawClosing() {
    const { context, width, height } = prepareCanvas();
    const n = Number(controls.labN.value);
    const winding = Number(controls.labWinding.value);
    context.fillStyle = plotColours.text;
    if (!(winding >= 3) || 2 * winding >= n) {
      context.textAlign = "center";
      context.fillText("Right-angle circles: k = sin(2πj/N), μ = π/2.", width / 2, height / 2);
      controls.analysisHint.textContent =
        "For m = N/2 the curvature satisfies κₙ₊₂ = −κₙ and the circles close at " +
        "k = sin(2πj/N), 1 ≤ j < N/4. Choose m < N/2 in the Model lab to see Θ(k).";
      return;
    }
    let curve;
    try {
      curve = closingCurve(n, winding);
    } catch (error) {
      context.textAlign = "center";
      context.fillText(String(error.message || error), width / 2, height / 2);
      return;
    }
    const roots = Families.closingModuli(n, winding);
    const left = 34;
    const right = width - 8;
    const top = 10;
    const bottom = height - 20;
    const kMax = curve.moduli[curve.moduli.length - 1];
    const thetaMin = -winding * Math.PI;
    const toScreen = (k, theta) => [
      left + (k / kMax) * (right - left),
      top + (theta / thetaMin) * (bottom - top),
    ];
    context.strokeStyle = plotColours.axis;
    context.strokeRect(left, top, right - left, bottom - top);
    context.textAlign = "right";
    for (let level = 0; level <= Math.floor(winding / 2); level += 1) {
      const [, y] = toScreen(0, -2 * Math.PI * level);
      context.strokeStyle = level === 0 ? plotColours.axis : "rgba(241, 188, 106, 0.4)";
      context.setLineDash(level === 0 ? [] : [3, 3]);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
      context.fillStyle = plotColours.text;
      context.fillText(level === 0 ? "0" : `−${2 * level}π`, left - 3, y + 3);
    }
    context.setLineDash([]);
    context.fillText(`−${winding}π`, left - 3, bottom);
    context.strokeStyle = plotColours.lime;
    context.lineWidth = 1.6;
    context.beginPath();
    curve.moduli.forEach((k, index) => {
      const [x, y] = toScreen(k, curve.theta[index]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
    context.lineWidth = 1;
    const active = currentCircle && currentCircle.n === n &&
      currentCircle.family.winding === winding ? currentCircle.level : 0;
    roots.forEach((k, index) => {
      const [x, y] = toScreen(k, -2 * Math.PI * (index + 1));
      context.fillStyle = index + 1 === active ? plotColours.amber : plotColours.ink;
      context.beginPath();
      context.arc(x, y, index + 1 === active ? 4 : 3, 0, 2 * Math.PI);
      context.fill();
    });
    context.fillStyle = plotColours.text;
    context.textAlign = "left";
    context.fillText(`Θ(k), N = ${n}, m = ${winding}`, left + 4, bottom - 5);
    context.textAlign = "right";
    context.fillText(`k → ${kMax.toFixed(3)} (c = 1)`, right, height - 5);
    controls.analysisHint.textContent =
      "Closing angle of the frame monodromy about the conserved axis. It decreases " +
      "strictly from 0 to −mπ; each crossing of −2πj is a closed critical circle " +
      `(${roots.length} for this N and m). The highlighted dot is the current model.`;
  }

  function drawAnalysis() {
    if (!model) return;
    if (analysisMode === "portrait") drawPortrait();
    else if (analysisMode === "closing") drawClosing();
    else drawProfile();
  }

  function setAnalysisMode(mode) {
    analysisMode = mode;
    Object.entries(controls.analysisButtons).forEach(([key, button]) => {
      button.classList.toggle("active", key === mode);
      button.setAttribute("aria-selected", String(key === mode));
    });
    drawAnalysis();
  }

  // ---------------------------------------------------------------------
  // Model description
  // ---------------------------------------------------------------------

  function inertiaType(inertia) {
    if (!Array.isArray(inertia)) return null;
    const [positive, negative] = inertia;
    if (negative === 0) return "local minimum of μ";
    if (positive === 0) return "local maximum of μ";
    return `saddle of index ${negative}`;
  }

  function updateAbout() {
    const metadata = modelMetadata || {};
    controls.description.textContent = metadata.description || "";
    controls.description.hidden = !metadata.description;
    const chips = [];
    const chip = (text, kind = "") => chips.push([text, kind]);
    chip(`N = ${model.n}`);
    chip(model.sign === 1 ? "oriented · σ = +1" : "anti-oriented · σ = −1");
    const family = metadata.family_parameters;
    if (family?.family === "cn") {
      chip(`m = ${family.m}${family.mirror ? "*" : ""}`);
      chip(`j = ${family.j}`);
      chip(`k = ${Number(family.k).toFixed(6)}`);
    } else if (family?.family === "torus_knot") {
      chip(`cells ${family.cells} · turns ${family.turns}`);
    }
    const inertia = metadata.hessian_inertia;
    if (Array.isArray(inertia)) {
      chip(`inertia (${inertia.join(", ")})`);
      chip(inertiaType(inertia), "accent");
    }
    if (metadata.knot) chip(`knot ${metadata.knot}`, "accent");
    if (metadata.kind === "counterexample") chip("counterexample", "warn");
    controls.chips.replaceChildren(...chips.map(([text, kind]) => {
      const span = document.createElement("span");
      span.className = `chip ${kind}`.trim();
      span.textContent = text;
      return span;
    }));
  }

  // ---------------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------------

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
    const result = areaNormal
      .multiplyScalar(0.86)
      .addScaledVector(tangent, 0.38)
      .addScaledVector(side, 0.18)
      .normalize();
    if (result.y < 0) {
      result.multiplyScalar(-1);
    }
    result.y += 0.16;
    result.normalize();
    return result;
  }

  function resetCamera() {
    const centred = displayedPoints || (model ? model.displayVertices() : [[0, 0, 0]]);
    const radius = Math.max(2, ...centred.map((point) => Math.hypot(...point)));
    const view = preferredViewDirection(centred);
    cameraState.yaw = Math.atan2(view.x, view.z);
    cameraState.pitch = Math.asin(THREE.MathUtils.clamp(view.y, -1, 1));
    // Fit the ring horizontally as well, for narrow (portrait) viewports.
    const halfHeight = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const halfWidth = Math.max(0.05, camera.aspect) * halfHeight;
    cameraState.distance = Math.max(7.5, radius * 3.65, (radius * 1.35) / halfWidth);
    cameraState.target.set(0, 0, 0);
    floor.position.y = -Math.max(3.25, radius * 1.08);
    grid.position.y = floor.position.y + 0.02;
    updateCamera();
  }

  // ---------------------------------------------------------------------
  // Flow controls
  // ---------------------------------------------------------------------

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
    controls.flowRateValue.textContent = `${Number(controls.flowRate.value).toFixed(2)}×`;
    controls.flowHint.textContent = hierarchy
      ? "The flows X(j) commute and conserve E₁, E₂, μ and closure. On a critical " +
        "circle X(1) moves exactly along the phase circle."
      : "The semi-discrete sine–Gordon flow commutes with the hierarchy; it is " +
        "defined for anti-oriented cycles.";
    element("flowStatus").textContent = running
      ? `${direction < 0 ? "−" : "+"} ${flowLabel()}`
      : "Paused";
  }

  function updatePhaseControl() {
    const enabled = Boolean(currentCircle);
    controls.phase.disabled = !enabled;
    if (!enabled) {
      controls.phaseValue.textContent = "—";
      return;
    }
    const period = currentCircle.family.phasePeriod;
    const fraction = ((currentPhase / period) % 1 + 1) % 1;
    controls.phase.value = String(fraction);
    controls.phaseValue.textContent = `ω = ${fraction.toFixed(3)} · 4K`;
  }

  // ---------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------

  function circleFromMetadata(metadata) {
    const family = metadata?.family_parameters;
    if (family?.family !== "cn") return null;
    try {
      return Families.criticalCircle(
        Number(family.N),
        Number(family.m_base ?? family.m),
        Number(family.j),
        { mirror: Boolean(family.mirror) },
      );
    } catch (error) {
      console.warn("Could not rebuild the critical circle.", error);
      return null;
    }
  }

  function loadData(data, name, sampleKey = "", options = {}) {
    try {
      setRunning(false);
      model = new Mathematics.Model(data);
      modelMetadata = { ...(data.metadata || {}) };
      currentSampleKey = sampleKey;
      currentFileName = name.replace(/\.json$/i, "") || model.name;
      currentCircle = options.circle === undefined
        ? circleFromMetadata(modelMetadata)
        : options.circle;
      currentPhase = options.phase ?? Number(modelMetadata.family_parameters?.phase ?? 0);
      generatedSource = options.generated || null;
      displayRotation = IDENTITY.slice();
      displayReference = null;
      displayedPoints = null;
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
      if (generatedSource) {
        setGeneratedOption(modelMetadata.label || model.name);
        controls.sample.value = GENERATED_KEY;
      } else {
        setGeneratedOption(null);
        controls.sample.value = sampleKey;
      }
      element("configurationTag").textContent = `N = ${model.n} · ${
        generatedSource ? "generated" : sampleKey ? sampleSource : "file"
      }`;
      element("cycleStatus").textContent =
        `K${model.n} · ${model.sign === 1 ? "oriented" : "anti-oriented"}`;
      element("editState").textContent = "Ready";
      pointerState.vertex = -1;
      pointerState.hovered = -1;
      updateFlowControls();
      updatePhaseControl();
      updateAbout();
      rebuildGeometry();
      resetCamera();
      writeHash();
      return true;
    } catch (error) {
      console.error(error);
      toast(error instanceof Error ? error.message : String(error));
      return false;
    }
  }

  function loadSample(key) {
    if (!samples[key]) return false;
    return loadData(samples[key], key, key);
  }

  // ---------------------------------------------------------------------
  // Model lab
  // ---------------------------------------------------------------------

  function populateWindings() {
    const n = Number(controls.labN.value);
    controls.labNValue.textContent = String(n);
    const previous = Number(controls.labWinding.value) || 3;
    controls.labWinding.replaceChildren();
    for (let m = 3; 2 * m <= n; m += 1) {
      const option = document.createElement("option");
      option.value = String(m);
      option.textContent = m === 3 && 2 * m < n
        ? "3 · principal"
        : 2 * m === n ? `${m} · right angle` : String(m);
      controls.labWinding.append(option);
    }
    controls.labWinding.value = String(Math.min(previous, Math.floor(n / 2)));
    if (!controls.labWinding.value) controls.labWinding.value = "3";
    populateLevels();
  }

  function populateLevels() {
    const n = Number(controls.labN.value);
    const m = Number(controls.labWinding.value);
    const previous = Number(controls.labLevel.value) || 1;
    const count = Families.closingLevelCount(n, m);
    controls.labLevel.replaceChildren();
    for (let level = 1; level <= count; level += 1) {
      const option = document.createElement("option");
      option.value = String(level);
      option.textContent = level === 1 ? "1 · once round" : String(level);
      controls.labLevel.append(option);
    }
    controls.labLevel.value = String(Math.min(previous, count));
    updateLabInfo();
  }

  function populateKnotControls() {
    if (!controls.labCells.children.length) {
      for (let cells = 3; cells <= 15; cells += 1) {
        const option = document.createElement("option");
        option.value = String(cells);
        option.textContent = `${cells} · N = ${3 * cells}`;
        controls.labCells.append(option);
      }
      controls.labCells.value = "5";
    }
    const cells = Number(controls.labCells.value);
    const previous = Number(controls.labTurns.value) || 2;
    controls.labTurns.replaceChildren();
    for (let turns = 1; 2 * turns < cells; turns += 1) {
      if (Families.gcd(cells, turns) !== 1) continue;
      const option = document.createElement("option");
      option.value = String(turns);
      option.textContent = turns === 1 ? "1 · unknot" : `${turns} · knotted`;
      controls.labTurns.append(option);
    }
    const values = [...controls.labTurns.options].map((option) => Number(option.value));
    controls.labTurns.value = String(values.includes(previous) ? previous : values.at(-1));
    updateLabInfo();
  }

  function labCircle() {
    return Families.criticalCircle(
      Number(controls.labN.value),
      Number(controls.labWinding.value),
      Number(controls.labLevel.value),
      { mirror: controls.labMirror.checked },
    );
  }

  function updateLabInfo() {
    controls.labInfo.classList.remove("error");
    try {
      if (labMode === "knot") {
        const cells = Number(controls.labCells.value);
        const turns = Number(controls.labTurns.value);
        const mirror = controls.labKnotMirror.checked;
        const selfLinking = mirror ? 1.5 * cells - cells * turns : cells * turns;
        const sign = mirror && cells % 2 === 1 ? "−1" : "+1";
        const knot = turns === 1 ? "unknot" : `T(${turns},${cells}) torus knot`;
        controls.labInfo.textContent =
          `N = ${3 * cells} · σ = ${sign} · ${knot} · SL = ${selfLinking}`;
        return;
      }
      const circle = labCircle();
      const inertia = circle.inertia
        ? `inertia (${circle.inertia.join(", ")}): ${inertiaType(circle.inertia)}`
        : "one-parameter turning motion";
      controls.labInfo.textContent =
        `k = ${circle.family.modulus.toFixed(6)} · μ = ${(circle.torsionAngle / Math.PI).toFixed(5)}π · ` +
        `σ = ${circle.sign === 1 ? "+1" : "−1"} · ${inertia}`;
    } catch (error) {
      controls.labInfo.textContent = error instanceof Error ? error.message : String(error);
      controls.labInfo.classList.add("error");
    }
    if (analysisMode === "closing") drawAnalysis();
  }

  function setLabMode(mode) {
    labMode = mode;
    controls.labCriticalTab.classList.toggle("active", mode === "critical");
    controls.labKnotTab.classList.toggle("active", mode === "knot");
    controls.labCriticalTab.setAttribute("aria-selected", String(mode === "critical"));
    controls.labKnotTab.setAttribute("aria-selected", String(mode === "knot"));
    controls.labCritical.hidden = mode !== "critical";
    controls.labKnot.hidden = mode !== "knot";
    controls.labPrincipal.hidden = mode !== "critical";
    if (mode === "knot") populateKnotControls();
    else updateLabInfo();
  }

  function criticalData(circle, phase) {
    const n = circle.n;
    const m = circle.family.winding;
    const j = circle.level;
    const star = circle.mirror ? "mirror " : "";
    const label = circle.rightAngle
      ? `Right-angle ${star}circle · N = ${n}, j = ${j}`
      : m === 3 && j === 1 && !circle.mirror
        ? `Principal family · N = ${n}`
        : `cn ${star}circle · N = ${n}, m = ${m}, j = ${j}`;
    const type = circle.inertia ? inertiaType(circle.inertia) : "the turning motion of K6";
    const description = [
      `Exact torsion-critical cn circle κₙ = A cn(ω + nδ; k), δ = 2Km/N, generated in the browser.`,
      circle.mirror
        ? "Mirror dual of the m-family: μ* = π − μ, winding N − m."
        : j === 1
          ? "Closing index j = 1: the centre line goes once around the conserved axis, so it is an embedded unknot with SL = m/2."
          : `Closing index j = ${j}: the centre line winds ${j} times around the conserved axis and can be knotted.`,
      `Torsion Hessian: ${type}.`,
    ].join(" ");
    return {
      name: `cn_n${n}_m${m}_j${j}${circle.mirror ? "_mirror" : ""}`,
      n,
      hinges: circle.hingesAt(phase),
      metadata: {
        kind: "critical",
        label,
        description,
        oriented: circle.sign === 1,
        family_parameters: {
          family: "cn",
          N: n,
          m: circle.winding,
          m_base: m,
          j,
          k: circle.family.modulus,
          mirror: circle.mirror,
          phase,
          phase_period: circle.family.phasePeriod,
        },
        hessian_inertia: circle.inertia,
        generated_in: "Kaleidocycle Studio",
      },
    };
  }

  function generateCritical(n, m, j, mirror, phaseFraction = null) {
    const circle = Families.criticalCircle(n, m, j, { mirror });
    const phase = phaseFraction === null ? 0.3 : phaseFraction * circle.family.phasePeriod;
    const data = criticalData(circle, phase);
    return loadData(data, data.name, "", {
      circle,
      phase,
      generated: { type: "critical", n, m, j, mirror },
    });
  }

  function generateKnot(cells, turns, mirror) {
    const n = 3 * cells;
    const selfLinking = mirror ? 1.5 * cells - cells * turns : cells * turns;
    const knot = turns === 1 ? null : `T(${turns},${cells})`;
    const data = {
      name: `torus_knot_m${cells}_k${turns}${mirror ? "_mirror" : ""}`,
      n,
      hinges: Families.torusKnotHinges(cells, turns, { mirror }),
      metadata: {
        kind: "knot",
        label: `${mirror ? "Mirror torus" : "Torus"}-knot K${n} · ${knot || "unknot"}`,
        description:
          `Explicit ring with ${cells}-fold rotational symmetry built from one ` +
          `three-hinge cell (turns = ${turns}). ${mirror
            ? "Mirror dual: B[n] ↦ (−1)ⁿB[n]."
            : "Oriented, with every curvature positive."} Its centre line is ` +
          `${knot ? `the ${knot} torus knot` : "an unknot"} and SL = ${selfLinking}, ` +
          "so the self-linking can grow quadratically in N.",
        oriented: !(mirror && cells % 2 === 1),
        family_parameters: { family: "torus_knot", cells, turns, mirror },
        knot: knot ? `${knot}${mirror ? " (mirror image)" : ""}` : undefined,
        generated_in: "Kaleidocycle Studio",
      },
    };
    return loadData(data, data.name, "", {
      circle: null,
      generated: { type: "knot", cells, turns, mirror },
    });
  }

  function generateFromLab() {
    try {
      if (labMode === "knot") {
        generateKnot(
          Number(controls.labCells.value),
          Number(controls.labTurns.value),
          controls.labKnotMirror.checked,
        );
      } else {
        generateCritical(
          Number(controls.labN.value),
          Number(controls.labWinding.value),
          Number(controls.labLevel.value),
          controls.labMirror.checked,
        );
      }
    } catch (error) {
      console.error(error);
      toast(error instanceof Error ? error.message : String(error));
    }
  }

  function mirrorModel() {
    if (!model) return;
    const hingesNow = Families.mirrorHinges(model.hinges());
    const metadata = {
      kind: modelMetadata.kind,
      label: modelMetadata.label?.startsWith("Mirror of ")
        ? modelMetadata.label.slice("Mirror of ".length)
        : `Mirror of ${modelMetadata.label || model.name}`,
      description:
        "Mirror duality B[n] ↦ (−1)ⁿB[n] of the previous configuration: " +
        "μ ↦ π − μ, σ ↦ (−1)ᴺσ and SL ↦ N/2 − SL.",
      generated_in: "Kaleidocycle Studio",
    };
    if (Array.isArray(modelMetadata.hessian_inertia)) {
      const [positive, negative, zero] = modelMetadata.hessian_inertia;
      metadata.hessian_inertia = [negative, positive, zero];
    }
    if (modelMetadata.knot) {
      metadata.knot = modelMetadata.knot.endsWith(" (mirror image)")
        ? modelMetadata.knot.replace(" (mirror image)", "")
        : `${modelMetadata.knot} (mirror image)`;
    }
    if (modelMetadata.family_parameters?.family === "torus_knot") {
      metadata.family_parameters = {
        ...modelMetadata.family_parameters,
        mirror: !modelMetadata.family_parameters.mirror,
      };
    }
    let circle = null;
    if (currentCircle) {
      circle = Families.criticalCircle(
        currentCircle.n,
        currentCircle.family.winding,
        currentCircle.level,
        { mirror: !currentCircle.mirror },
      );
      metadata.family_parameters = {
        ...(modelMetadata.family_parameters || {}),
        family: "cn",
        N: circle.n,
        m: circle.winding,
        m_base: circle.family.winding,
        j: circle.level,
        mirror: circle.mirror,
        phase: currentPhase,
      };
    }
    const mirrored = !(generatedSource?.mirror || currentSampleMirror);
    const source = generatedSource
      ? { ...generatedSource, mirror: !generatedSource.mirror }
      : null;
    const sampleKey = currentSampleKey;
    const ok = loadData(
      { name: `${model.name}_mirror`, n: model.n, hinges: hingesNow, metadata },
      `${currentFileName}_mirror`,
      "",
      { circle, phase: currentPhase, generated: source },
    );
    if (ok && !source && sampleKey) {
      currentSampleKey = sampleKey;
      currentSampleMirror = mirrored;
      if (mirrored) {
        setGeneratedOption(metadata.label);
        controls.sample.value = GENERATED_KEY;
      } else {
        controls.sample.value = sampleKey;
        modelMetadata = { ...(samples[sampleKey]?.metadata || modelMetadata) };
        updateAbout();
      }
      writeHash();
    }
    if (ok) toast("Mirror dual: μ ↦ π − μ, σ ↦ (−1)ᴺσ, SL ↦ N/2 − SL.");
  }

  let currentSampleMirror = false;

  // ---------------------------------------------------------------------
  // URL state
  // ---------------------------------------------------------------------

  function writeHash() {
    let hash = "";
    if (generatedSource?.type === "critical") {
      const { n, m, j, mirror } = generatedSource;
      const fraction = currentCircle
        ? ((currentPhase / currentCircle.family.phasePeriod) % 1 + 1) % 1
        : 0;
      hash = `critical=${n},${m},${j}${mirror ? "&mirror=1" : ""}&phase=${fraction.toFixed(4)}`;
    } else if (generatedSource?.type === "knot") {
      const { cells, turns, mirror } = generatedSource;
      hash = `knot=${cells},${turns}${mirror ? "&mirror=1" : ""}`;
    } else if (currentSampleKey) {
      hash = `sample=${encodeURIComponent(currentSampleKey)}${currentSampleMirror ? "&mirror=1" : ""}`;
    }
    try {
      const url = new URL(window.location.href);
      url.hash = hash;
      window.history.replaceState(null, "", url);
    } catch (error) {
      // History updates are a convenience; ignore failures (e.g. sandboxed frames).
    }
  }

  function applyHash() {
    const hash = window.location.hash.replace(/^#/, "");
    if (!hash) return false;
    const parameters = new URLSearchParams(hash);
    const mirror = parameters.get("mirror") === "1";
    try {
      if (parameters.has("critical")) {
        const [n, m, j] = parameters.get("critical").split(",").map(Number);
        controls.labN.value = String(n);
        populateWindings();
        controls.labWinding.value = String(m);
        populateLevels();
        controls.labLevel.value = String(j);
        controls.labMirror.checked = mirror;
        setLabMode("critical");
        const phase = Number(parameters.get("phase"));
        return generateCritical(n, m, j, mirror, Number.isFinite(phase) ? phase : null);
      }
      if (parameters.has("knot")) {
        const [cells, turns] = parameters.get("knot").split(",").map(Number);
        setLabMode("knot");
        controls.labCells.value = String(cells);
        populateKnotControls();
        controls.labTurns.value = String(turns);
        controls.labKnotMirror.checked = mirror;
        updateLabInfo();
        return generateKnot(cells, turns, mirror);
      }
      if (parameters.has("sample")) {
        const key = parameters.get("sample");
        if (!loadSample(key)) return false;
        currentSampleMirror = false;
        if (mirror) mirrorModel();
        return true;
      }
    } catch (error) {
      console.warn("Ignoring an invalid link.", error);
      toast("The link describes a model that could not be built.");
    }
    return false;
  }

  // ---------------------------------------------------------------------
  // Pointer interaction
  // ---------------------------------------------------------------------

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
    if (event.button !== 0 || !model) return;
    element("gestureHint").style.opacity = "0";
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
        // Handles are shown in display coordinates; edit in model coordinates.
        pendingDragTarget = rotateBack(displayRotation, target.toArray());
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
        Math.min(60, cameraState.distance * Math.exp(event.deltaY * 0.0012)),
      );
      updateCamera();
    },
    { passive: false },
  );

  // ---------------------------------------------------------------------
  // Files, capture and sharing
  // ---------------------------------------------------------------------

  controls.sample.addEventListener("change", () => {
    const key = controls.sample.value;
    if (key === GENERATED_KEY) return;
    currentSampleMirror = false;
    loadSample(key);
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
        currentSampleMirror = false;
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

  function download(blob, filename) {
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }

  controls.export.addEventListener("click", () => {
    const payload = model.toJSON();
    payload.metadata = { ...payload.metadata, ...modelMetadata, ...{
      name: payload.metadata.name,
      n: payload.metadata.n,
      oriented: payload.metadata.oriented,
      edited_in: "Kaleidocycle Studio",
    } };
    download(
      new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
      `${currentFileName}_edited.json`,
    );
    toast("The current constrained configuration was exported.");
  });

  if (Print) {
    const defaults = Print.DEFAULTS;
    controls.printWidth.value = String(defaults.ribbon_width);
    controls.printEdge.value = String(defaults.edge_length);
    controls.printThickness.value = String(defaults.body_thickness);
    controls.printHinge.value = defaults.hinge_type;
    controls.printRing.checked = defaults.with_ring;
    controls.printPinFit.value = String(defaults.pin_fit_clearance);
    controls.printPinTurn.value = String(defaults.pin_clearance);
    controls.printEdgeGap.value = String(defaults.gap);
    const settingInputs = [
      controls.printWidth, controls.printEdge, controls.printThickness, controls.printHinge,
      controls.printRing, controls.printPinFit, controls.printPinTurn,
      controls.printKnuckleGap, controls.printEdgeGap,
    ];
    settingInputs.forEach((input) => input.addEventListener("change", () => {
      printState.contacts.clear();
      printState.closestState = null;
      controls.printClosest.disabled = true;
      if (printState.enabled) schedulePrintBuild(60);
      else setPrintTag(printState.unit ? "Ready" : "Off");
    }));
    controls.printShow.addEventListener("change", () => setPrintEnabled(controls.printShow.checked));
    controls.printPlate.addEventListener("click", exportPrintPlate);
    controls.printLink.addEventListener("click", exportPrintLink);
    controls.printAssembly.addEventListener("click", exportPrintAssembly);
    controls.printCheck.addEventListener("click", checkPrintContacts);
    controls.printClosest.addEventListener("click", showClosestPose);
  } else {
    controls.printCard.hidden = true;
  }

  controls.capture.addEventListener("click", () => {
    renderer.render(scene, camera);
    const anchor = document.createElement("a");
    anchor.download = `${currentFileName}.png`;
    anchor.href = renderer.domElement.toDataURL("image/png");
    anchor.click();
  });

  function toggleRecording() {
    if (recorder) {
      recorder.stop();
      return;
    }
    if (!renderer.domElement.captureStream || typeof MediaRecorder === "undefined") {
      toast("Video recording is not supported by this browser.");
      return;
    }
    const types = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4",
    ];
    const mimeType = types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
    const chunks = [];
    try {
      recorder = new MediaRecorder(
        renderer.domElement.captureStream(60),
        mimeType ? { mimeType, videoBitsPerSecond: 8e6 } : undefined,
      );
    } catch (error) {
      recorder = null;
      toast("Could not start the recorder.");
      return;
    }
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      const type = recorder.mimeType || mimeType || "video/webm";
      const extension = type.includes("mp4") ? "mp4" : "webm";
      download(new Blob(chunks, { type }), `${currentFileName}.${extension}`);
      recorder = null;
      controls.record.classList.remove("recording");
      controls.record.setAttribute("aria-pressed", "false");
      controls.record.title = "Record a video clip";
      toast("The clip was saved.");
    });
    recorder.start(250);
    controls.record.classList.add("recording");
    controls.record.setAttribute("aria-pressed", "true");
    controls.record.title = "Stop recording";
    toast("Recording… press ● again to stop.");
  }
  controls.record.addEventListener("click", toggleRecording);

  controls.copyLink.addEventListener("click", async () => {
    writeHash();
    const link = window.location.href;
    try {
      await navigator.clipboard.writeText(link);
      toast("A link to this model was copied.");
    } catch (error) {
      window.prompt("Copy this link:", link);
    }
  });

  function setAutoRotate(next) {
    autoRotate = Boolean(next);
    controls.autoRotate.classList.toggle("active", autoRotate);
    controls.autoRotate.setAttribute("aria-pressed", String(autoRotate));
  }
  controls.autoRotate.addEventListener("click", () => setAutoRotate(!autoRotate));

  function toggleHelp(force) {
    const show = force ?? controls.helpOverlay.hidden;
    controls.helpOverlay.hidden = !show;
    controls.help.setAttribute("aria-expanded", String(show));
    controls.help.classList.toggle("active", show);
  }
  controls.help.addEventListener("click", () => toggleHelp());

  // ---------------------------------------------------------------------
  // Panel controls
  // ---------------------------------------------------------------------

  function stepFlow() {
    setRunning(false);
    const requested = direction * Number(controls.flowRate.value) * 0.0012;
    model.advanceFlow(
      controls.flowKind.value,
      Number(controls.flowOrder.value),
      requested,
    );
    model.project(3);
    rebuildGeometry();
  }

  function reverseFlow() {
    direction *= -1;
    controls.reverse.textContent = direction < 0 ? "Forward" : "Reverse";
    updateFlowControls();
  }

  function resetShape() {
    setRunning(false);
    model.restore();
    controls.torsion.value = String(model.torsionAngle);
    currentPhase = Number(modelMetadata.family_parameters?.phase ?? currentPhase);
    updatePhaseControl();
    rebuildGeometry();
    toast("The initial configuration was restored.");
  }

  controls.play.addEventListener("click", () => setRunning(!running));
  controls.reverse.addEventListener("click", reverseFlow);
  controls.step.addEventListener("click", stepFlow);
  controls.flowKind.addEventListener("change", updateFlowControls);
  controls.flowOrder.addEventListener("input", updateFlowControls);
  controls.flowRate.addEventListener("input", updateFlowControls);

  controls.torsion.addEventListener("input", () => {
    setRunning(false);
    pendingTorsion = Number(controls.torsion.value);
    controls.torsionValue.textContent = `${pendingTorsion.toFixed(4)} rad`;
    element("editState").textContent = "Solving…";
  });
  controls.resetShape.addEventListener("click", resetShape);
  controls.setInitial.addEventListener("click", () => {
    model.initialState = model.snapshot();
    model.flowTime = 0;
    updateDiagnostics();
    toast("The current shape is now the initial configuration.");
  });
  controls.mirror.addEventListener("click", mirrorModel);

  controls.labCriticalTab.addEventListener("click", () => setLabMode("critical"));
  controls.labKnotTab.addEventListener("click", () => setLabMode("knot"));
  controls.labN.addEventListener("input", populateWindings);
  controls.labWinding.addEventListener("change", populateLevels);
  controls.labLevel.addEventListener("change", updateLabInfo);
  controls.labMirror.addEventListener("change", updateLabInfo);
  controls.labCells.addEventListener("change", populateKnotControls);
  controls.labTurns.addEventListener("change", updateLabInfo);
  controls.labKnotMirror.addEventListener("change", updateLabInfo);
  controls.labGenerate.addEventListener("click", generateFromLab);
  controls.labPrincipal.addEventListener("click", () => {
    if (Number(controls.labN.value) < 7) controls.labN.value = "7";
    populateWindings();
    controls.labWinding.value = "3";
    populateLevels();
    controls.labLevel.value = "1";
    controls.labMirror.checked = false;
    updateLabInfo();
    generateFromLab();
  });
  controls.phase.addEventListener("input", () => {
    if (!currentCircle) return;
    setRunning(false);
    pendingPhase = Number(controls.phase.value);
  });
  controls.phase.addEventListener("change", writeHash);

  Object.entries(controls.analysisButtons).forEach(([mode, button]) =>
    button.addEventListener("click", () => setAnalysisMode(mode)));

  controls.colourMode.addEventListener("change", rebuildGeometry);
  controls.ribbonWidth.addEventListener("input", () => {
    controls.widthValue.textContent = Number(controls.ribbonWidth.value).toFixed(2);
    rebuildGeometry();
  });
  controls.opacity.addEventListener("input", () => {
    controls.opacityValue.textContent = Number(controls.opacity.value).toFixed(2);
    surfaceMaterial.opacity = Number(controls.opacity.value);
  });
  [controls.centreline, controls.hinges, controls.handles, controls.stabilise].forEach(
    (control) => control.addEventListener("change", rebuildGeometry),
  );
  controls.resetView.addEventListener("click", resetCamera);

  window.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || !model) return;
    const target = event.target;
    const tag = target?.tagName;
    if (tag === "SELECT" || tag === "TEXTAREA") return;
    if (tag === "INPUT" && target.type !== "range" && target.type !== "checkbox") return;
    const key = event.key;
    if (key === " " && tag !== "BUTTON" && target.type !== "checkbox") {
      event.preventDefault();
      setRunning(!running);
    } else if (key === ".") stepFlow();
    else if (key === "b" || key === "B") reverseFlow();
    else if (key === "[" || key === "]") {
      const next = Number(controls.flowOrder.value) + (key === "]" ? 1 : -1);
      controls.flowOrder.value = String(
        Math.max(1, Math.min(Number(controls.flowOrder.max), next)),
      );
      controls.flowKind.value = "hierarchy";
      updateFlowControls();
    } else if (key === "r" || key === "R") resetShape();
    else if (key === "m" || key === "M") mirrorModel();
    else if (key === "v" || key === "V") resetCamera();
    else if (key === "a" || key === "A") setAutoRotate(!autoRotate);
    else if ((key === "p" || key === "P") && Print) setPrintEnabled(!printState.enabled);
    else if (key === "?") toggleHelp();
    else if (key === "Escape") toggleHelp(false);
  });

  window.addEventListener("hashchange", () => {
    // Ignore our own replaceState updates; react to edited links only.
    applyHash();
  });

  function resizeRenderer() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
  }
  const resizeObserver = new ResizeObserver(() => {
    resizeRenderer();
    drawAnalysis();
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
    if (pendingPhase !== null && currentCircle) {
      const fraction = pendingPhase;
      pendingPhase = null;
      currentPhase = fraction * currentCircle.family.phasePeriod;
      try {
        model.setHinges(currentCircle.hingesAt(currentPhase));
        controls.torsion.value = String(model.torsionAngle);
        controls.phaseValue.textContent = `ω = ${fraction.toFixed(3)} · 4K`;
        element("editState").textContent = "On the critical circle";
      } catch (error) {
        console.error(error);
      }
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
    if (autoRotate && pointerState.mode !== "orbit" && elapsed > 0) {
      cameraState.yaw += elapsed * 0.32;
      updateCamera();
    }
    renderer.render(scene, camera);
  }

  async function start() {
    try {
      await loadSampleData();
      resizeRenderer();
      populateSamples();
      populateWindings();
      populateKnotControls();
      setLabMode("critical");
      updateFlowControls();
      if (!applyHash()) {
        const initialKey = controls.sample.value;
        loadSample(initialKey);
      }
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
