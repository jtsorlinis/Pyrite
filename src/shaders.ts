// WGSL for the two passes:
//   cull: one thread per (cluster, instance). Runs the parallel DAG-cut LOD test plus frustum
//         culling, and appends survivors to a visible list + indirect draw args.
//   draw: vertex pulling. One instance per visible cluster, 128*3 vertices per instance;
//         triangles past the cluster's count collapse to nothing.

const common = /* wgsl */ `
struct Uniforms {
  viewProj: mat4x4f,
  planes: array<vec4f, 6>,
  camPos: vec4f,
  lodScale: f32,      // viewport height in CSS pixels / (2 * tan(fovY / 2))
  threshold: f32,     // max allowed projected error, CSS pixels
  minDist: f32,
  spacing: f32,       // instance grid spacing
  numClusters: u32,
  gridSize: u32,
  viewMode: u32,      // 0 = cluster, 1 = triangle, 2 = plain
  capacity: u32,      // size of the visible list
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
  pad0: u32,
  pad1: u32,
  levels: array<atomic<u32>, 32>,
};

@group(0) @binding(2) var<storage, read_write> drawArgs: DrawArgs;
@group(0) @binding(3) var<storage, read_write> visible: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> stats: Stats;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let ci = gid.x;
  let inst = gid.y;

  var draw = false;
  var triCount = 0u;
  var level = 0u;
  if (ci < u.numClusters) {
    let c = clusters[ci];
    let off = instanceOffset(inst);
    // The Nanite cut: draw me if I'm accurate enough but my parent group isn't.
    // Both sides use the exact same (sphere, error) values stored on parent and child,
    // so siblings and cousins always agree and the cut never has holes or overlaps.
    let ownOk = projectedError(c.lodBounds, c.lodError, off) <= u.threshold;
    let parentTooCoarse = c.parentLevel == ROOT || projectedError(c.parentBounds, c.parentError, off) > u.threshold;
    draw = ownOk && parentTooCoarse && inFrustum(c.bounds, off);
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

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) ids: vec2u, // cluster, triangle
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
  return out;
}

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

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // Flat face normal from screen-space derivatives (no vertex normals needed).
  var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
  let v = normalize(u.camPos.xyz - in.world);
  if (dot(n, v) < 0.0) {
    n = -n;
  }

  var base: vec3f;
  switch u.viewMode {
    case 0u: { base = idColor(in.ids.x * 7919u + 1u); }
    case 1u: { base = idColor(in.ids.x * 131071u + in.ids.y * 7u + 3u); }
    default: { base = vec3f(0.75, 0.74, 0.72); }
  }
  let albedo = pow(base, vec3f(2.2));

  let sunDir = normalize(vec3f(0.45, 0.8, 0.35));
  let diffuse = max(dot(n, sunDir), 0.0);
  let hemi = mix(vec3f(0.05, 0.045, 0.04), vec3f(0.12, 0.14, 0.18), n.y * 0.5 + 0.5);
  let head = max(dot(n, v), 0.0);
  let color = albedo * (hemi + diffuse * vec3f(0.95, 0.9, 0.82) + head * 0.18);
  return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
`;
