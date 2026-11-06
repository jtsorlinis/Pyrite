import { Renderer, type GpuStats } from './renderer';
import { OrbitCamera } from './camera';
import { lookAt, perspectiveReversedInfinite, type Vec3 } from './math';
import { loadGltf, type MaterialInfo } from './loader';
import type { BuildResult, MeshInput } from './builder';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type Source = { kind: 'file'; file: File } | { kind: 'url'; url: string };

const DEFAULT_MODEL = 'models/dragon.glb';

const state = {
  viewMode: 0,
  nanite: true, // off = draw the original mesh with no LOD
  threshold: 1, // max projected error in pixels while Nanite is on
  budget: 5_000_000, // triangle cap, 0 = off: the GPU raises the error just enough to fit it
  instances: 1,
};

const canvas = $<HTMLCanvasElement>('gpu');
let renderer: Renderer;
let camera: OrbitCamera;
let model: BuildResult | null = null;
let modelName = '';
let materialSummary = { maps: '', sizes: '', bytes: 0, textured: false };

main();

async function main() {
  try {
    renderer = await Renderer.create(canvas);
  } catch (e) {
    const el = $('fatal');
    el.hidden = false;
    el.innerHTML = `<div><b>WebGPU unavailable</b><br>${(e as Error).message}<br><span class="muted">Try a recent Chrome, Edge or Safari.</span></div>`;
    return;
  }
  camera = new OrbitCamera(canvas);
  renderer.onStats = onStats;
  bindUi();
  const files = await listProjectModels();

  const param = new URLSearchParams(location.search).get('model');
  const initial = param ?? (files.includes(DEFAULT_MODEL) ? DEFAULT_MODEL : files[0]);
  if (initial) loadModel({ kind: 'url', url: initial });
  else $('model-info').innerHTML = '<dt>Open or drop a .glb to start</dt><dd></dd>';

  requestAnimationFrame(frame);
}

// --- frame loop --------------------------------------------------------------------------------

let lastT = performance.now();
let frameMs = 16;

function frame(t: number) {
  frameMs += ((t - lastT) - frameMs) * 0.05;
  lastT = t;

  const radius = model?.radius ?? 1;
  const near = Math.max(Math.min(camera.distance * 0.01, radius * 0.01), radius * 1e-6);
  const proj = perspectiveReversedInfinite(camera.fovY, renderer.aspect, near);
  const eye = camera.eye;
  const grid = gridDims();
  renderer.render({
    view: lookAt(eye, camera.target, [0, 1, 0]),
    proj,
    near,
    camPos: eye,
    fovY: camera.fovY,
    nanite: state.nanite,
    threshold: state.threshold,
    budget: state.budget,
    viewMode: state.viewMode,
    gridCols: grid.cols,
    gridRows: grid.rows,
    spacing: spacing(),
  });
  requestAnimationFrame(frame);
}

function spacing() {
  if (!model) return 1;
  return Math.max(model.aabbMax[0] - model.aabbMin[0], model.aabbMax[2] - model.aabbMin[2], model.radius * 0.5) * 1.25;
}

/** The instance grid: the most nearly square cols x rows that holds exactly state.instances. */
function gridDims() {
  const n = state.instances;
  let rows = Math.floor(Math.sqrt(n));
  while (n % rows) rows--;
  return { cols: n / rows, rows };
}

/** Fits the camera to every instance, using the model's vertices so nothing is fitted loosely. */
function reframe() {
  if (!model) return;
  const { cols, rows } = gridDims();
  const hx = (spacing() * (cols - 1)) / 2;
  const hz = (spacing() * (rows - 1)) / 2;
  const corners: Vec3[] =
    state.instances === 1
      ? [[0, 0, 0]]
      : [[-hx, 0, -hz], [hx, 0, -hz], [-hx, 0, hz], [hx, 0, hz]];
  // A lone model gets a little more room around it than a grid does.
  camera.fit(model.positions, corners, renderer.aspect, state.instances === 1 ? 0.8 : 0.9);
}

// --- model loading -----------------------------------------------------------------------------

let jobId = 0;
let worker: Worker | null = null;
let cancelBuild: (() => void) | null = null;

function build(req: MeshInput, transfer: Transferable[]): Promise<BuildResult> {
  cancelBuild?.();
  worker?.terminate();
  const w = (worker = new Worker(new URL('./builder.worker.ts', import.meta.url), { type: 'module' }));
  return new Promise((resolve, reject) => {
    cancelBuild = () => reject(new Error('cancelled'));
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') setProgress(m.stage, m.fraction);
      else if (m.type === 'done') resolve(m.result);
      else if (m.type === 'error') reject(new Error(m.message));
    };
    w.onerror = (e) => reject(new Error(e.message || 'Worker failed'));
    w.postMessage(req, transfer);
  });
}

async function loadModel(src: Source) {
  const my = ++jobId;
  const label = displayName(src.kind === 'file' ? src.file.name : src.url);
  showOverlay(`Loading ${label}`);
  try {
    setProgress('Reading file', 0);
    const buf = src.kind === 'file' ? await src.file.arrayBuffer() : await fetchWithProgress(src.url);
    if (my !== jobId) return;
    setProgress('Parsing glTF', 0.01);
    await new Promise((r) => setTimeout(r, 30)); // let the overlay paint
    const mesh = await loadGltf(buf);
    if (my !== jobId) return;
    // Textures decode on the main thread while the worker builds the DAG.
    const { positions, normals, uvs, indices } = mesh;
    const result = await build({ positions, normals, uvs, indices }, [positions.buffer, normals.buffer, uvs.buffer, indices.buffer]);
    if (my !== jobId) return;
    setProgress('Decoding textures', 1);
    const material = await mesh.material;
    if (my !== jobId) return;
    materialSummary = summarizeMaterial(material, mesh.materialCount);
    renderer.setScene(result, material, mesh.doubleSided);
    model = result;
    modelName = label;
    onModelLoaded();
    hideOverlay();
    const url = new URL(location.href);
    if (src.kind === 'file') url.searchParams.delete('model');
    else url.searchParams.set('model', src.url);
    history.replaceState(null, '', url);
  } catch (e) {
    if (my !== jobId || (e as Error).message === 'cancelled') return;
    console.error(e);
    showError(`Couldn't load ${label}`, (e as Error).message);
  }
}

async function fetchWithProgress(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  if ((res.headers.get('content-type') ?? '').includes('text/html')) throw new Error(`${url} not found`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return res.arrayBuffer();
  // Content-Length is the size on the wire. When the server compresses the response (GitHub Pages
  // gzips .glb files) the body we read is bigger than that, so the buffer grows as needed and the
  // progress shows no total.
  const compressed = res.headers.has('content-encoding');
  const mb = (n: number) => (n / 2 ** 20).toFixed(0);
  let out = new Uint8Array(total);
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > out.length) {
      const bigger = new Uint8Array(Math.max(out.length * 2, got + value.length));
      bigger.set(out.subarray(0, got));
      out = bigger;
    }
    out.set(value, got);
    got += value.length;
    setProgress(compressed ? `Downloading ${mb(got)} MB` : `Downloading ${mb(got)} / ${mb(total)} MB`, 0);
  }
  return got === out.length ? out.buffer : out.buffer.slice(0, got);
}

async function listProjectModels(): Promise<string[]> {
  try {
    // Relative, so it resolves next to index.html wherever the build is hosted.
    const res = await fetch('models.json');
    if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) return [];
    const files: string[] = await res.json();
    const box = $('models');
    for (const f of files) {
      const b = document.createElement('button');
      b.className = 'file';
      b.textContent = displayName(f);
      b.title = `Load ${f}`;
      b.onclick = () => loadModel({ kind: 'url', url: f });
      box.insertBefore(b, $('btn-open'));
    }
    return files;
  } catch {
    return []; // no model list: open or drop a file instead
  }
}

/** One line for the model panel, plus texture memory (with mips) for the GPU total. */
function summarizeMaterial(mat: MaterialInfo | null, count: number) {
  if (!mat) return { maps: `${count} materials (unsupported)`, sizes: '', bytes: 0, textured: false };
  const slots: [string, ImageBitmap | null][] = [
    ['colour', mat.baseColor],
    ['normal', mat.normal],
    ['rough', mat.metallicRoughness],
  ];
  const present = slots.filter(([, b]) => b);
  const bytes = present.reduce((sum, [, b]) => sum + (b!.width * b!.height * 4 * 4) / 3, 0);
  const size = (w: number) => (w >= 1024 ? `${+(w / 1024).toFixed(1)}K` : `${w}px`);
  const sizes = [...new Set(present.map(([, b]) => size(b!.width)))].join('/');
  const maps = present.length ? present.map(([n]) => n).join(', ') : 'none';
  return { maps, sizes, bytes, textured: true };
}

function onModelLoaded() {
  const m = model!;
  let dagTris = 0;
  for (const l of m.levels) dagTris += l.triangles;
  const geometryBytes = m.indices.byteLength + m.positions.byteLength + m.normals.byteLength + m.uvs.byteLength + m.clusters.byteLength;
  const mb = (bytes: number) => `${(bytes / 2 ** 20).toFixed(0)} MB`;
  const rows = [
    ['Model', modelName],
    ['Source triangles', fmt(m.sourceTriangles)],
    ['Welded vertices', fmt(m.positions.length / 3)],
    ['DAG', `${fmt(m.clusterCount)} clusters · ${m.levels.length} levels`],
    ['DAG triangles', `${fmt(dagTris)} (${(dagTris / m.sourceTriangles).toFixed(2)}×)`],
    ['Mesh memory', mb(geometryBytes)],
    ['Textures', materialSummary.maps],
  ];
  if (materialSummary.bytes) rows.push(['Texture memory', `${mb(materialSummary.bytes)} · ${materialSummary.sizes}`]);
  rows.push(['Build time', `${(m.buildMs / 1000).toFixed(2)} s`]);
  $('model-info').innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`)
    .join('');

  const texButton = $('view-mode').querySelector<HTMLButtonElement>('[data-v="0"]')!;
  texButton.disabled = !materialSummary.textured;
  texButton.title = materialSummary.textured ? '' : 'Texturing supports models with a single material';
  if (texButton.disabled && state.viewMode === 0) $('view-mode').querySelector<HTMLButtonElement>('[data-v="1"]')!.click();

  const bars = $('levels');
  bars.innerHTML = '';
  m.levels.forEach((l, i) => {
    const d = document.createElement('div');
    d.title = `Level ${i}: ${fmt(l.clusters)} clusters, ${fmt(l.triangles)} triangles in DAG`;
    bars.appendChild(d);
  });
  reframe();
}

// --- stats -------------------------------------------------------------------------------------

let lastStatsUpdate = 0;
let gpuCull = 0;
let gpuDraw = 0;
function onStats(s: GpuStats) {
  const now = performance.now();
  if (now - lastStatsUpdate < 120 || !model) return;
  lastStatsUpdate = now;
  const sceneTris = model.sourceTriangles * state.instances;
  $('st-tris').textContent = fmt(s.triangles);
  const drawn = fmt(Math.min(s.clusters, renderer.capacity));
  $('st-clusters').textContent = `${drawn} (${fmt(s.lateClusters)} late)`;
  $('st-occ').textContent = `${fmt(s.occluded)} clusters`;
  const t = s.threshold;
  const px = `${t < 1 ? t.toFixed(2) : t.toFixed(1)} px`;
  $('st-err').textContent = !state.nanite ? 'none (full mesh)' : t > 1e30 ? 'coarsest (over cap)' : s.capped ? `${px} (capped)` : px;
  $('st-scene').textContent = compact(sceneTris);
  const ratio = s.triangles / sceneTris;
  $('st-ratio').textContent = s.triangles ? `${(ratio * 100).toFixed(ratio < 0.01 ? 3 : 1)}%  (1 : ${compact(Math.round(1 / ratio))})` : '–';
  $('st-fps').textContent = `${frameMs.toFixed(1)} ms · ${Math.round(1000 / frameMs)} fps`;
  if (s.gpuCullMs !== null && s.gpuDrawMs !== null) {
    gpuCull += (s.gpuCullMs - gpuCull) * 0.3;
    gpuDraw += (s.gpuDrawMs - gpuDraw) * 0.3;
    $('st-gpu').textContent = `${gpuCull.toFixed(2)} ms + ${gpuDraw.toFixed(2)} ms`;
  }

  const bars = $('levels').children;
  let max = 1;
  for (let i = 0; i < bars.length; i++) max = Math.max(max, s.levels[i]);
  for (let i = 0; i < bars.length; i++) {
    const el = bars[i] as HTMLElement;
    const n = s.levels[i];
    el.style.height = `${n ? Math.max(6, (100 * n) / max) : 4}%`;
    el.dataset.n = String(n);
    el.title = `Level ${i}: ${fmt(n)} clusters drawn (${fmt(model.levels[i]?.clusters ?? 0)} in DAG)`;
  }
  const of = $('overflow');
  of.hidden = !s.overflow;
  const fix = state.nanite ? 'Lower the quality or turn on the triangle cap.' : 'Enable Nanite or use fewer instances.';
  if (s.overflow) of.textContent = `Visible list full: ${fmt(s.overflow)} clusters dropped. ${fix}`;
}

// --- UI ----------------------------------------------------------------------------------------

function bindUi() {
  const fileInput = $<HTMLInputElement>('file');
  $('btn-open').onclick = () => fileInput.click();
  fileInput.onchange = () => {
    const f = fileInput.files?.[0];
    if (f) loadModel({ kind: 'file', file: f });
    fileInput.value = '';
  };

  const modes = segmented('view-mode', (v) => (state.viewMode = v));
  segmented('quality', (v) => (state.threshold = v));
  segmented('budget', (v) => (state.budget = v));
  segmented('instances', (v) => {
    state.instances = v;
    reframe();
  });
  const nanite = $<HTMLButtonElement>('nanite');
  nanite.onclick = () => {
    state.nanite = !state.nanite;
    nanite.classList.toggle('on', state.nanite);
    nanite.setAttribute('aria-pressed', String(state.nanite));
    nanite.textContent = state.nanite ? 'Nanite enabled' : 'Nanite disabled';
    $('lod-options').hidden = !state.nanite;
  };

  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === 'KeyR') reframe();
    else if (/^Digit[1-2]$/.test(e.code)) modes[Number(e.code.slice(5)) - 1].click();
  });

  // Drag & drop
  const drop = $('drop');
  let depth = 0;
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    depth++;
    drop.hidden = false;
  });
  window.addEventListener('dragleave', () => {
    if (--depth <= 0) {
      depth = 0;
      drop.hidden = true;
    }
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    drop.hidden = true;
    const f = e.dataTransfer?.files?.[0];
    if (f) loadModel({ kind: 'file', file: f });
  });
}

/** Wires up a row of toggle buttons; each button's data-v is the value it selects. */
function segmented(id: string, onChange: (v: number) => void) {
  const buttons = $(id).querySelectorAll<HTMLButtonElement>('button');
  buttons.forEach((b) => {
    b.onclick = () => {
      buttons.forEach((x) => x.classList.toggle('on', x === b));
      onChange(Number(b.dataset.v));
    };
  });
  return buttons;
}

function showOverlay(title: string) {
  $('overlay').hidden = false;
  $('ov-title').textContent = title;
  $('ov-detail').textContent = '';
  $('ov-fill').style.width = '0%';
  $('overlay').onclick = null;
}
function setProgress(stage: string, fraction: number) {
  $('ov-detail').textContent = stage;
  $('ov-fill').style.width = `${Math.round(fraction * 100)}%`;
}
function hideOverlay() {
  $('overlay').hidden = true;
}
function showError(title: string, detail: string) {
  showOverlay(title);
  $('ov-detail').textContent = `${detail} (click to dismiss)`;
  $('overlay').onclick = hideOverlay;
}

// --- formatting --------------------------------------------------------------------------------

/** "models/dragon.glb" -> "dragon" */
function displayName(path: string) {
  return path.split('/').pop()!.replace(/\.(glb|gltf)$/i, '');
}
function fmt(n: number) {
  return Math.round(n).toLocaleString('en-US');
}
function compact(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  return fmt(n);
}
function escapeHtml(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}
