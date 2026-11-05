// WGSL for the passes:
//   cull: three compute entry points, run in order each frame:
//         histogram - only with a triangle cap: for every visible (cluster, instance), how many
//                     triangles it adds to the cut at each pixel-error threshold (a histogram over
//                     log2(threshold))
//         pick      - the pixel-error setting, raised just enough to fit the cap if there is one
//         main      - the parallel DAG-cut LOD test with that threshold, plus frustum culling;
//                     appends survivors to a visible list + indirect draw args
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
  gridSize: u32,
  viewMode: u32,      // 0 = textured, 1 = triangle colours
  capacity: u32,      // size of the visible list
  baseColorFactor: vec4f,
  metallicFactor: f32,
  roughnessFactor: f32,
  normalScale: f32,
  fullDetail: u32,    // 1 = Nanite off: draw the original mesh (every level-0 cluster)
  budget: u32,        // triangle cap, 0 = none; pick raises the threshold until the cut fits
  pad0: u32,
  pad1: u32,
  pad2: u32,
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
  let n = u.gridSize;
  let half = f32(n - 1u) * 0.5;
  return vec3f(f32(i % n) - half, 0.0, f32(i / n) - half) * u.spacing;
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
};

struct LodState {
  threshold: f32,     // max projected error in rendered pixels, written by pick, read by main
};

@group(0) @binding(2) var<storage, read_write> drawArgs: DrawArgs;
@group(0) @binding(3) var<storage, read_write> visible: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> stats: Stats;
// hist[k] = change in drawn triangles when the threshold steps up to thresholdOf(k): + for clusters
// whose own error is reached there, - for clusters whose parent's error is reached (they hand over
// to the parent). Wrapping u32 maths: running totals are exact as long as the true count fits.
@group(0) @binding(5) var<storage, read_write> hist: array<atomic<u32>, BINS>;
@group(0) @binding(6) var<storage, read_write> lod: LodState;

// Thresholds from 1/64 px to 1024 px in steps of 2^(1/32) (about 2%).
const BINS = 512u;
const BINS_PER_OCTAVE = 32.0;
const LOG2_MIN = -6.0;
const NOTHING_FITS = 3.0e38; // even the coarsest cut is over budget: draw the DAG roots
const HIST_GROUP = 256u;     // histogram workgroup size: must match the dispatch in renderer.ts

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

// Pass 1. Cluster c is in the cut for thresholds t with own <= t < parent, so it adds triCount at
// bin(own) and removes it again at bin(parent). Bins are summed in workgroup memory first: every
// level-0 cluster lands in bin 0, and one global atomic per thread there would serialize badly.
// (Workgroup memory starts zeroed, as WGSL guarantees.)
@compute @workgroup_size(HIST_GROUP)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  let ci = gid.x;
  var a = BINS;
  var b = BINS;
  if (ci < u.numClusters) {
    let c = clusters[ci];
    let off = instanceOffset(gid.y);
    if (inFrustum(c.bounds, off)) {
      a = binOf(projectedError(c.lodBounds, c.lodError, off));
      if (c.parentLevel != ROOT) {
        b = binOf(projectedError(c.parentBounds, c.parentError, off));
      }
      // Own and parent error in the same bin: the cluster is never in the cut at any threshold we
      // can pick, so it adds nothing. That's most fine clusters of distant instances.
      if (a < b) {
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

// Pass 2. Start from the pixel-error setting. With a cap, also find the finest threshold whose cut
// fits it (triangles in the cut only go down as the threshold rises, so walk up from the finest and
// stop at the first that fits: 512 steps on one thread, a few microseconds) and use whichever of
// the two is coarser. The quality presets sit exactly on bins, so a setting that already fits is
// never raised.
@compute @workgroup_size(1)
fn pick() {
  var chosen = u.threshold;
  var capped = 0u;
  if (u.budget > 0u) {
    var total = 0u;
    var fits = NOTHING_FITS;
    for (var k = 0u; k < BINS; k++) {
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
  lod.threshold = chosen;
  stats.threshold = chosen;
  stats.capped = capped;
}

// Pass 3: select and append.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let ci = gid.x;
  let inst = gid.y;
  let threshold = lod.threshold;

  var draw = false;
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
    draw = selected && inFrustum(c.bounds, off);
    triCount = c.triCount;
    level = c.level;
  }

  // Append to the visible list with one global atomic per workgroup.
  var slot = 0u;
  if (draw) {
    slot = atomicAdd(&wgCount, 1u);
  }
  workgroupBarrier();
  if (lid == 0u) {
    let n = atomicLoad(&wgCount);
    if (n > 0u) {
      wgBase = atomicAdd(&drawArgs.instanceCount, n);
    }
  }
  let base = workgroupUniformLoad(&wgBase);
  if (draw) {
    let idx = base + slot;
    if (idx < u.capacity) {
      visible[idx] = vec2u(ci, inst);
      atomicAdd(&stats.tris, triCount);
      atomicAdd(&stats.levels[min(level, 31u)], 1u);
    } else {
      atomicAdd(&stats.overflow, 1u);
    }
  }
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
  if (ii >= u.capacity) {
    return out;
  }
  let entry = visible[ii];
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
