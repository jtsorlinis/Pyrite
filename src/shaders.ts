// WGSL for the passes:
//   cull: four compute entry points, run in order each frame:
//         histogram - only with a triangle cap: for every visible (cluster, instance), how many
//                     triangles it adds to the cut at each pixel-error threshold (a histogram over
//                     log2(threshold)). Visible as main sees it: in the frustum and not behind last
//                     frame's depth, so the cap counts what gets drawn.
//         pick      - the pixel-error setting, raised just enough to fit the cap if there is one
//         main      - the parallel DAG-cut LOD test with that threshold, plus frustum culling and
//                     (with occlusion culling) a test against last frame's HZB; appends what passes
//                     to the visible list for the first draw, and what the HZB hides to a deferred list
//         post      - only with occlusion culling, after the first draw and an HZB rebuilt from it:
//                     retests the deferred list against that HZB and appends what shows for a
//                     second draw. Anything hidden last frame but visible now is caught here.
//   hzb:  builds the depth pyramid (farthest depth per texel) from the MSAA depth buffer.
//   draw: vertex pulling. One instance per visible cluster, 128*3 vertices per instance;
//         triangles past the cluster's count collapse to nothing.
//   mip:  fullscreen blit used to build texture mip chains (WebGPU has no generateMipmap).

const common = /* wgsl */ `
struct Uniforms {
  viewProj: mat4x4f,
  planes: array<vec4f, 6>,
  camPos: vec4f,
  lodScale: f32,      // render-target height in pixels / (2 * tan(fovY / 2))
  threshold: f32,     // max projected error, rendered pixels (the quality setting)
  minDist: f32,
  spacing: f32,       // instance grid spacing
  numClusters: u32,
  gridCols: u32,      // instances are laid out on a gridCols x gridRows grid
  viewMode: u32,      // 0 = textured, 1 = triangle colours
  capacity: u32,      // size of the visible list
  baseColorFactor: vec4f,
  metallicFactor: f32,
  roughnessFactor: f32,
  normalScale: f32,
  fullDetail: u32,    // 1 = Nanite off: draw the original mesh (every level-0 cluster)
  budget: u32,        // triangle cap, 0 = none; pick raises the threshold until the cut fits
  gridRows: u32,
  pad1: u32,
  pad2: u32,
  view: mat4x4f,      // world -> view space (the camera looks down -z)
  prevView: mat4x4f,  // last frame's, which the HZB was rendered from when main reads it
  proj: vec4f,        // P[0][0], P[1][1], near plane, unused
  prevProj: vec4f,
  screen: vec2u,      // render-target size in pixels
  testPrev: u32,      // 1 = main tests against last frame's HZB (occlusion on, and the HZB is current)
  pad3: u32,
};

struct Cluster {
  bounds: vec4f,        // culling sphere of this cluster's triangles
  lodBounds: vec4f,     // sphere of the group that produced this cluster
  parentBounds: vec4f,  // sphere of the group this cluster was simplified into
  lodError: f32,
  parentError: f32,
  triOffset: u32,
  triCount: u32,
  level: u32,
  parentLevel: u32,     // 0xffffffff for roots
  parentGroup: u32,
  pad: u32,
};

const ROOT = 0xffffffffu;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;

fn instanceOffset(i: u32) -> vec3f {
  let n = u.gridCols;
  let half = vec2f(f32(n - 1u), f32(u.gridRows - 1u)) * 0.5;
  return vec3f(f32(i % n) - half.x, 0.0, f32(i / n) - half.y) * u.spacing;
}
`;

export const cullShader = /* wgsl */ `
${common}

struct DrawArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};

struct Stats {
  tris: atomic<u32>,
  overflow: atomic<u32>,
  threshold: f32,     // chosen by the pick pass, for the UI
  capped: u32,        // 1 if the triangle cap had to raise the threshold
  levels: array<atomic<u32>, 32>,
  occluded: atomic<u32>, // clusters post found hidden by this frame's depth
};

struct CullState {
  deferredCount: atomic<u32>, // pairs main found behind last frame's depth (may exceed capacity)
  threshold: f32,     // max projected error in rendered pixels, written by pick, read by main
};

// Workgroup count for post, written by main. Post's pipeline doesn't bind it: a buffer can't be
// both an indirect argument and writable storage in the same dispatch.
struct PostArgs {
  x: atomic<u32>,
  y: u32,
  z: u32,
};

// [0] = the first draw (main), [1] = the second (post). Their entries share the visible list:
// post's follow main's.
@group(0) @binding(2) var<storage, read_write> drawArgs: array<DrawArgs, 2>;
@group(0) @binding(3) var<storage, read_write> visible: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> stats: Stats;
// hist[k] = change in drawn triangles when the threshold steps up to thresholdOf(k): + for clusters
// whose own error is reached there, - for clusters whose parent's error is reached (they hand over
// to the parent). Wrapping u32 maths: running totals are exact as long as the true count fits.
@group(0) @binding(5) var<storage, read_write> hist: array<atomic<u32>, BINS>;
@group(0) @binding(6) var<storage, read_write> cullState: CullState;
@group(0) @binding(7) var<storage, read_write> deferred: array<vec2u>;
// Farthest depth per texel; level L texels cover 2^(L+1) pixels (see hzbShader).
@group(1) @binding(0) var hzb: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> postArgs: PostArgs; // main's layout only

// Thresholds from 1/64 px to 1024 px in steps of 2^(1/32) (about 2%).
const BINS = 512u;
const BINS_PER_OCTAVE = 32.0;
const LOG2_MIN = -6.0;
const NOTHING_FITS = 3.0e38; // even the coarsest cut is over budget: draw the DAG roots
const HIST_GROUP = 256u;     // histogram workgroup size: must match the dispatch in renderer.ts
const POST_GROUP = 128u;     // post workgroup size (main sizes post's indirect dispatch with it)

fn thresholdOf(k: u32) -> f32 {
  return exp2(LOG2_MIN + f32(k) / BINS_PER_OCTAVE);
}

// Smallest bin whose threshold is >= e (BINS if e is beyond the range).
fn binOf(e: f32) -> u32 {
  let k = ceil((log2(max(e, 1e-30)) - LOG2_MIN) * BINS_PER_OCTAVE);
  return u32(clamp(k, 0.0, f32(BINS)));
}

var<workgroup> localHist: array<atomic<u32>, BINS>;

var<workgroup> wgCount: atomic<u32>;
var<workgroup> wgBase: u32;
var<workgroup> wgDeferCount: atomic<u32>;
var<workgroup> wgDeferBase: u32;

// Error in pixels if a simplification error of 'err' were seen at the closest point of sphere 's'.
fn projectedError(s: vec4f, err: f32, off: vec3f) -> f32 {
  let d = max(distance(s.xyz + off, u.camPos.xyz) - s.w, u.minDist);
  return err * u.lodScale / d;
}

fn inFrustum(s: vec4f, off: vec3f) -> bool {
  let c = s.xyz + off;
  for (var i = 0u; i < 5u; i++) {
    let p = u.planes[i];
    if (dot(p.xyz, c) + p.w < -s.w) {
      return false;
    }
  }
  return true;
}

// True when sphere 's' (moved by 'off') is certainly behind every depth the HZB holds where it
// lands on screen, seen by the camera (view, proj). False when that can't be shown: the sphere
// crosses the near plane or misses the screen.
fn occluded(s: vec4f, off: vec3f, view: mat4x4f, proj: vec4f) -> bool {
  let v = view * vec4f(s.xyz + off, 1.0);
  let c = vec3f(v.xy, -v.z); // +z forward
  let r = s.w;
  let near = proj.z;
  if (c.z - r < near) {
    return false;
  }
  // Exact screen bounds of a perspective-projected sphere, from the planes through the eye that
  // touch it (Mara & McGuire 2013). The denominators stay positive while the sphere is in front.
  let czr2 = c.z * c.z - r * r;
  let vx = sqrt(c.x * c.x + czr2);
  let vy = sqrt(c.y * c.y + czr2);
  let minX = (vx * c.x - r * c.z) / (vx * c.z + r * c.x) * proj.x;
  let maxX = (vx * c.x + r * c.z) / (vx * c.z - r * c.x) * proj.x;
  let minY = (vy * c.y - r * c.z) / (vy * c.z + r * c.y) * proj.y;
  let maxY = (vy * c.y + r * c.z) / (vy * c.z - r * c.y) * proj.y;
  // NDC (y up) to pixels (y down).
  let size = vec2f(u.screen);
  let lo = (vec2f(minX, -maxY) * 0.5 + 0.5) * size;
  let hi = (vec2f(maxX, -minY) * 0.5 + 0.5) * size;
  if (any(hi < vec2f(0.0)) || any(lo >= size)) {
    return false;
  }
  let p0 = vec2u(clamp(lo, vec2f(0.0), size - 1.0));
  let p1 = vec2u(clamp(hi, vec2f(0.0), size - 1.0));

  // A level-L texel covers 2^(L+1) pixels, so at the level where that's at least the rect's size
  // the rect touches at most 2x2 texels.
  let extent = max(p1.x - p0.x, p1.y - p0.y) + 1u;
  let ceilLog2 = select(firstLeadingBit(extent - 1u) + 1u, 0u, extent <= 1u);
  let level = min(max(ceilLog2, 1u) - 1u, textureNumLevels(hzb) - 1u);
  let last = textureDimensions(hzb, level) - 1u;
  let shift = vec2u(level + 1u);
  let t0 = min(p0 >> shift, last); // a level 1 texel wide along an axis covers all of it
  let t1 = min(p1 >> shift, last);
  let farthest = min(
    min(textureLoad(hzb, t0, level).r, textureLoad(hzb, vec2u(t1.x, t0.y), level).r),
    min(textureLoad(hzb, vec2u(t0.x, t1.y), level).r, textureLoad(hzb, t1, level).r),
  );
  // Reversed-Z with an infinite far plane: depth = near / distance, so bigger means closer.
  let nearest = near / (c.z - r);
  return nearest < farthest;
}

// Appends this thread's (cluster, instance) to the visible list of draw 'which', after 'start'
// entries already there, with one global atomic per workgroup. Every thread of the workgroup must
// call it.
fn appendVisible(draw: bool, which: u32, start: u32, entry: vec2u, triCount: u32, level: u32, lid: u32) {
  var slot = 0u;
  if (draw) {
    slot = atomicAdd(&wgCount, 1u);
  }
  workgroupBarrier();
  if (lid == 0u) {
    let n = atomicLoad(&wgCount);
    if (n > 0u) {
      wgBase = atomicAdd(&drawArgs[which].instanceCount, n);
    }
  }
  let base = workgroupUniformLoad(&wgBase);
  if (draw) {
    let idx = start + base + slot;
    if (idx < u.capacity) {
      visible[idx] = entry;
      atomicAdd(&stats.tris, triCount);
      atomicAdd(&stats.levels[min(level, 31u)], 1u);
    } else {
      atomicAdd(&stats.overflow, 1u);
    }
  }
}

// Pass 1. Cluster c is in the cut for thresholds t with own <= t < parent, so it adds triCount at
// bin(own) and removes it again at bin(parent). Pick never goes finer than the quality setting, so
// counting starts at that bin: a cluster the cut has left by then adds nothing. Bins are summed in
// workgroup memory first: many clusters land in the same bin, and one global atomic per thread
// there would serialize badly. (Workgroup memory starts zeroed, as WGSL guarantees.)
@compute @workgroup_size(HIST_GROUP)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  let ci = gid.x;
  var a = BINS;
  var b = BINS;
  if (ci < u.numClusters) {
    let c = clusters[ci];
    let off = instanceOffset(gid.y);
    if (inFrustum(c.bounds, off)) {
      a = max(binOf(projectedError(c.lodBounds, c.lodError, off)), binOf(u.threshold));
      if (c.parentLevel != ROOT) {
        b = binOf(projectedError(c.parentBounds, c.parentError, off));
      }
      // a >= b: the cluster is never in the cut at any threshold pick can choose, so it adds
      // nothing. That's most fine clusters. What last frame's depth hides, main defers and post
      // only draws if this frame's depth shows it, which a still camera's never does, so it isn't
      // counted. (A moving camera can bring a few of those into view and run slightly over.)
      if (a < b && !(u.testPrev != 0u && occluded(c.bounds, off, u.prevView, u.prevProj))) {
        atomicAdd(&localHist[a], c.triCount);
        if (b < BINS) {
          atomicSub(&localHist[b], c.triCount);
        }
      } else {
        a = BINS;
        b = BINS;
      }
    }
  }
  workgroupBarrier();
  // Flush only the bins this workgroup touched: whichever thread swaps a bin out first carries
  // its total to the global histogram, the rest find zero.
  if (a < BINS) {
    let n = atomicExchange(&localHist[a], 0u);
    if (n != 0u) {
      atomicAdd(&hist[a], n);
    }
  }
  if (b < BINS) {
    let n = atomicExchange(&localHist[b], 0u);
    if (n != 0u) {
      atomicAdd(&hist[b], n);
    }
  }
}

// Pass 2. Start from the pixel-error setting. With a cap, also find the finest threshold from
// there up whose cut fits it (triangles in the cut only go down as the threshold rises, so walk up
// and stop at the first that fits: at most 512 steps on one thread, a few microseconds) and use
// whichever of the two is coarser. The quality presets sit exactly on bins, so a setting that
// already fits is never raised. Histogram counted nothing below the setting's bin.
@compute @workgroup_size(1)
fn pick() {
  var chosen = u.threshold;
  var capped = 0u;
  if (u.budget > 0u) {
    var total = 0u;
    var fits = NOTHING_FITS;
    for (var k = binOf(u.threshold); k < BINS; k++) {
      total += atomicLoad(&hist[k]);
      if (total <= u.budget) {
        fits = thresholdOf(k);
        break;
      }
    }
    if (fits > chosen) {
      chosen = fits;
      capped = 1u;
    }
  }
  cullState.threshold = chosen;
  stats.threshold = chosen;
  stats.capped = capped;
}

// Pass 3: select, cull, append.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let ci = gid.x;
  let inst = gid.y;
  let threshold = cullState.threshold;

  var draw = false;
  var late = false;
  var triCount = 0u;
  var level = 0u;
  if (ci < u.numClusters) {
    let c = clusters[ci];
    let off = instanceOffset(inst);
    // The Nanite cut: draw me if I'm accurate enough but my parent group isn't.
    // Both sides use the exact same (sphere, error) values stored on parent and child,
    // so siblings and cousins always agree and the cut never has holes or overlaps.
    var selected: bool;
    if (u.fullDetail != 0u) {
      selected = c.level == 0u;
    } else {
      let ownOk = projectedError(c.lodBounds, c.lodError, off) <= threshold;
      let parentTooCoarse = c.parentLevel == ROOT || projectedError(c.parentBounds, c.parentError, off) > threshold;
      selected = ownOk && parentTooCoarse;
    }
    if (selected && inFrustum(c.bounds, off)) {
      // Hidden behind last frame's depth: likely still hidden, so it waits for post, which checks
      // it against this frame's depth once the first draw is in.
      late = u.testPrev != 0u && occluded(c.bounds, off, u.prevView, u.prevProj);
      draw = !late;
    }
    triCount = c.triCount;
    level = c.level;
  }

  // Deferred pairs go to post's list, one global atomic per workgroup, growing post's dispatch.
  var deferSlot = 0u;
  if (late) {
    deferSlot = atomicAdd(&wgDeferCount, 1u);
  }
  workgroupBarrier();
  if (lid == 0u) {
    let n = atomicLoad(&wgDeferCount);
    if (n > 0u) {
      let b = atomicAdd(&cullState.deferredCount, n);
      wgDeferBase = b;
      atomicMax(&postArgs.x, (min(b + n, u.capacity) + POST_GROUP - 1u) / POST_GROUP);
    }
  }
  let deferBase = workgroupUniformLoad(&wgDeferBase);
  if (late) {
    let idx = deferBase + deferSlot;
    if (idx < u.capacity) {
      deferred[idx] = vec2u(ci, inst);
    } else {
      draw = true; // no room to defer it: draw it now rather than lose it
    }
  }

  appendVisible(draw, 0u, 0u, vec2u(ci, inst), triCount, level, lid);
}

// Pass 4, after the first draw and an HZB built from it: what main deferred, against this frame's
// depth. The first draw only holds real geometry seen from this camera, so whatever this culls is
// hidden for sure.
@compute @workgroup_size(POST_GROUP)
fn post(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = gid.x;
  var draw = false;
  var entry = vec2u(0u);
  var triCount = 0u;
  var level = 0u;
  if (i < min(atomicLoad(&cullState.deferredCount), u.capacity)) {
    entry = deferred[i];
    let c = clusters[entry.x];
    draw = !occluded(c.bounds, instanceOffset(entry.y), u.view, u.proj);
    if (!draw) {
      atomicAdd(&stats.occluded, 1u);
    }
    triCount = c.triCount;
    level = c.level;
  }
  appendVisible(draw, 1u, atomicLoad(&drawArgs[0].instanceCount), entry, triCount, level, lid);
}
`;

// Depth pyramid. Each texel holds the farthest depth (smallest value, reversed-Z) of the pixels it
// covers: level-L texel t covers pixels [t, t + 1) * 2^(L+1). Level 0 is half the render size
// rounded up to powers of two, so every level halves exactly; texels past the screen's edge hold
// 1.0 (nearest), which min ignores. A dispatch reduces 16x16 tiles four levels down in workgroup
// memory, so the whole pyramid takes a few dispatches instead of one per level.
export const hzbShader = /* wgsl */ `
@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
@group(0) @binding(1) var src: texture_2d<f32>; // the last level the previous dispatch wrote
@group(0) @binding(2) var out0: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var out1: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var out2: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var out3: texture_storage_2d<r32float, write>;

const T = 16u;
var<workgroup> tile: array<f32, 256>;

// Stores past a level's size are skipped. (Levels the pyramid doesn't have are bound to 1x1
// stand-ins, which take the writes harmlessly.)
fn put(i: u32, p: vec2u, v: f32) {
  let c = vec4f(v, 0.0, 0.0, 0.0);
  switch i {
    case 0u: { if (all(p < textureDimensions(out0))) { textureStore(out0, p, c); } }
    case 1u: { if (all(p < textureDimensions(out1))) { textureStore(out1, p, c); } }
    case 2u: { if (all(p < textureDimensions(out2))) { textureStore(out2, p, c); } }
    default: { if (all(p < textureDimensions(out3))) { textureStore(out3, p, c); } }
  }
}

// Halves the workgroup's 16x16 tile 'count' times (2x2 min each), storing step s with
// put(first + s - 1). 'v' is this thread's texel of the tile, whose first texel is 'origin'.
fn reduceTile(v: f32, local: vec2u, origin: vec2u, first: u32, count: u32) {
  tile[local.y * T + local.x] = v;
  workgroupBarrier();
  for (var s = 1u; s <= count; s++) {
    let inside = all(local < vec2u(T >> s));
    var m = 1.0;
    if (inside) {
      let i = 2u * (local.y * T + local.x);
      m = min(min(tile[i], tile[i + 1u]), min(tile[i + T], tile[i + T + 1u]));
      put(first + s - 1u, (origin >> vec2u(s)) + local, m);
    }
    workgroupBarrier();
    if (inside) {
      tile[local.y * T + local.x] = m;
    }
    workgroupBarrier();
  }
}

// Levels 0-3, from every sample of the MSAA depth buffer.
@compute @workgroup_size(16, 16)
fn fromDepth(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let screen = textureDimensions(depth);
  let samples = textureNumSamples(depth);
  var m = 1.0;
  for (var i = 0u; i < 4u; i++) {
    let p = gid.xy * 2u + vec2u(i & 1u, i >> 1u);
    if (all(p < screen)) {
      for (var s = 0u; s < samples; s++) {
        m = min(m, textureLoad(depth, p, s));
      }
    }
  }
  put(0u, gid.xy, m);
  reduceTile(m, lid.xy, wid.xy * T, 1u, 3u);
}

// The next four levels, from the last one written.
@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  var m = 1.0;
  if (all(gid.xy < textureDimensions(src))) {
    m = textureLoad(src, gid.xy, 0).r;
  }
  reduceTile(m, lid.xy, wid.xy * T, 0u, 4u);
}
`;

export const drawShader = /* wgsl */ `
${common}

@group(0) @binding(2) var<storage, read> visible: array<vec2u>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> positions: array<f32>;
@group(0) @binding(5) var<storage, read> normals: array<f32>;
@group(0) @binding(6) var<storage, read> uvs: array<f32>;
@group(0) @binding(7) var samp: sampler;
@group(0) @binding(8) var baseColorTex: texture_2d<f32>;
@group(0) @binding(9) var normalTex: texture_2d<f32>;
@group(0) @binding(10) var metalRoughTex: texture_2d<f32>;

struct DrawArgs {
  vertexCount: u32,
  instanceCount: u32,
  firstVertex: u32,
  firstInstance: u32,
};
@group(0) @binding(11) var<storage, read> drawArgs: array<DrawArgs, 2>;

// The second (post) draw's clusters follow the first's in the visible list. (WebGPU needs an
// optional feature for a non-zero firstInstance in an indirect draw, so the offset is applied here.)
override POST: bool = false;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) ids: vec2u, // cluster, triangle
  @location(2) normal: vec3f,
  @location(3) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  out.pos = vec4f(2.0, 2.0, 2.0, 1.0); // outside the clip volume: discarded
  let slot = select(ii, drawArgs[0].instanceCount + ii, POST);
  if (slot >= u.capacity) {
    return out;
  }
  let entry = visible[slot];
  let c = clusters[entry.x];
  let tri = vi / 3u;
  if (tri >= c.triCount) {
    return out;
  }
  let v = indices[(c.triOffset + tri) * 3u + vi % 3u];
  let p = vec3f(positions[v * 3u], positions[v * 3u + 1u], positions[v * 3u + 2u]) + instanceOffset(entry.y);
  out.pos = u.viewProj * vec4f(p, 1.0);
  out.world = p;
  out.ids = vec2u(entry.x, tri);
  out.normal = vec3f(normals[v * 3u], normals[v * 3u + 1u], normals[v * 3u + 2u]);
  out.uv = vec2f(uvs[v * 2u], uvs[v * 2u + 1u]);
  return out;
}

const PI = 3.14159265;
const SUN_DIR = vec3f(0.4558, 0.8103, 0.3546); // normalize(0.45, 0.8, 0.35)

fn hash(x: u32) -> u32 {
  var h = x;
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

fn hsv(h: f32, s: f32, v: f32) -> vec3f {
  let p = abs(fract(vec3f(h) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return v * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), s);
}

fn idColor(id: u32) -> vec3f {
  let h = hash(id);
  return hsv(f32(h & 0xffffu) / 65535.0, 0.45 + 0.4 * f32((h >> 16u) & 0xffu) / 255.0, 0.7 + 0.3 * f32(h >> 24u) / 255.0);
}

// Triangle view: flat-shaded facets, one colour per triangle, so LOD changes are visible.
fn shadeFlat(in: VSOut, base: vec3f) -> vec3f {
  var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
  let v = normalize(u.camPos.xyz - in.world);
  if (dot(n, v) < 0.0) {
    n = -n;
  }
  let albedo = pow(base, vec3f(2.2));
  let diffuse = max(dot(n, SUN_DIR), 0.0);
  let hemi = mix(vec3f(0.05, 0.045, 0.04), vec3f(0.12, 0.14, 0.18), n.y * 0.5 + 0.5);
  let head = max(dot(n, v), 0.0);
  return albedo * (hemi + diffuse * vec3f(0.95, 0.9, 0.82) + head * 0.18);
}

// Tangent frame from screen-space derivatives ("normal mapping without precomputed tangents"),
// so the mesh needs no tangent attribute and simplification can't break one.
fn perturbNormal(n: vec3f, p: vec3f, uv: vec2f, tn: vec3f) -> vec3f {
  let dp1 = dpdx(p);
  let dp2 = dpdy(p);
  let duv1 = dpdx(uv);
  let duv2 = dpdy(uv);
  let dp2perp = cross(dp2, n);
  let dp1perp = cross(n, dp1);
  let t = dp2perp * duv1.x + dp1perp * duv2.x;
  let b = dp2perp * duv1.y + dp1perp * duv2.y;
  let invmax = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-30));
  return normalize(mat3x3f(t * invmax, b * invmax, n) * tn);
}

// glTF metallic-roughness: Lambert diffuse + GGX specular from one sun, plus hemisphere ambient.
fn shadeTextured(in: VSOut, front: bool) -> vec3f {
  let base = textureSample(baseColorTex, samp, in.uv) * u.baseColorFactor; // linear (sRGB texture)
  let mr = textureSample(metalRoughTex, samp, in.uv);
  var tn = textureSample(normalTex, samp, in.uv).xyz * 2.0 - 1.0;
  // glTF normal maps are +Y up in the image, but our frame's bitangent points along +v (image down).
  tn = vec3f(tn.x * u.normalScale, -tn.y * u.normalScale, tn.z);

  var ng = normalize(in.normal);
  if (!front) {
    ng = -ng; // double-sided: light the back of a face from its own side
  }
  let n = perturbNormal(ng, in.world, in.uv, tn);
  let v = normalize(u.camPos.xyz - in.world);
  let rough = clamp(mr.g * u.roughnessFactor, 0.045, 1.0);
  let metal = clamp(mr.b * u.metallicFactor, 0.0, 1.0);

  let h = normalize(SUN_DIR + v);
  let nl = max(dot(n, SUN_DIR), 0.0);
  let nv = max(dot(n, v), 1e-4);
  let nh = max(dot(n, h), 0.0);
  let vh = max(dot(v, h), 0.0);
  let a2 = pow(rough, 4.0);
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let d = a2 / (PI * dd * dd);
  let k = (rough + 1.0) * (rough + 1.0) / 8.0;
  let g = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
  let f0 = mix(vec3f(0.04), base.rgb, metal);
  let f = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
  let spec = d * g * f / max(4.0 * nl * nv, 1e-4);
  let diffuse = (1.0 - f) * (1.0 - metal) * base.rgb / PI;
  let sun = vec3f(1.0, 0.96, 0.9) * 3.2;

  let sky = mix(vec3f(0.07, 0.06, 0.05), vec3f(0.2, 0.23, 0.28), n.y * 0.5 + 0.5);
  let ambient = sky * (base.rgb * (1.0 - metal) + f0 * (1.0 - rough * 0.7));
  let fill = base.rgb * (1.0 - metal) * nv * 0.12; // soft headlight so the shadow side keeps detail
  let c = (diffuse + spec) * sun * nl + ambient + fill;
  return c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14); // ACES (Narkowicz fit)
}

@fragment
fn fs(in: VSOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var color: vec3f;
  if (u.viewMode == 0u) {
    color = shadeTextured(in, front);
  } else {
    color = shadeFlat(in, idColor(in.ids.x * 131071u + in.ids.y * 7u + 3u));
  }
  return vec4f(pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)), 1.0);
}
`;

export const mipShader = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); // fullscreen triangle
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return textureSampleLevel(src, samp, in.uv, 0.0);
}
`;
