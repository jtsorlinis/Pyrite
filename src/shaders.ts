// WGSL for the two passes:
//   cull: one thread per (cluster, instance). Runs the parallel DAG-cut LOD test plus frustum
//         culling, and appends survivors to a visible list + indirect draw args.
//   draw: vertex pulling. One instance per visible cluster, 128*3 vertices per instance;
//         triangles past the cluster's count collapse to nothing.
//   sky:  (in the draw module) the environment map behind textured models.
//   mip:  fullscreen blit used to build texture mip chains (WebGPU has no generateMipmap).
//   prefilter: GGX-blurs the environment once per roughness level for reflections.

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
  viewMode: u32,      // 0 = cluster, 1 = triangle, 2 = plain, 3 = textured
  capacity: u32,      // size of the visible list
  baseColorFactor: vec4f,
  metallicFactor: f32,
  roughnessFactor: f32,
  normalScale: f32,
  pad: f32,
  invViewProj: mat4x4f,
  sh: array<vec4f, 9>,  // environment irradiance / PI, spherical harmonics
  exposure: f32,
  envMaxLod: f32,       // roughness 1 maps to this reflection mip
  pad1: f32,
  pad2: f32,
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
@group(0) @binding(5) var<storage, read> normals: array<f32>;
@group(0) @binding(6) var<storage, read> uvs: array<f32>;
@group(0) @binding(7) var samp: sampler;
@group(0) @binding(8) var baseColorTex: texture_2d<f32>;
@group(0) @binding(9) var normalTex: texture_2d<f32>;
@group(0) @binding(10) var metalRoughTex: texture_2d<f32>;
@group(0) @binding(11) var envSampler: sampler;
@group(0) @binding(12) var envSpecular: texture_2d<f32>; // mip i = GGX roughness i / envMaxLod
@group(0) @binding(13) var envSky: texture_2d<f32>;

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

// Debug views: flat-shaded facets so each triangle and cluster is visible.
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

/** Equirectangular lookup; matches texelDir() in environment.ts. */
fn equirectUv(d: vec3f) -> vec2f {
  return vec2f(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

fn irradiance(n: vec3f) -> vec3f {
  let s = u.sh;
  let e = s[0].xyz * 0.282095
    + s[1].xyz * (0.488603 * n.y) + s[2].xyz * (0.488603 * n.z) + s[3].xyz * (0.488603 * n.x)
    + s[4].xyz * (1.092548 * n.x * n.y) + s[5].xyz * (1.092548 * n.y * n.z)
    + s[6].xyz * (0.315392 * (3.0 * n.z * n.z - 1.0)) + s[7].xyz * (1.092548 * n.x * n.z)
    + s[8].xyz * (0.546274 * (n.x * n.x - n.y * n.y));
  return max(e, vec3f(0.0));
}

// Split-sum environment BRDF, analytic fit (Karis, "Physically Based Shading on Mobile").
fn envBrdf(f0: vec3f, rough: f32, nv: f32) -> vec3f {
  let r = rough * vec4f(-1.0, -0.0275, -0.572, 0.022) + vec4f(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}

fn tonemap(c: vec3f) -> vec3f {
  return c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14); // ACES (Narkowicz fit)
}

// glTF metallic-roughness lit only by the environment map: diffuse from its spherical harmonics,
// reflections from the GGX-prefiltered mip matching the surface roughness.
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

  let nv = max(dot(n, v), 1e-4);
  let f0 = mix(vec3f(0.04), base.rgb, metal);
  let brdf = envBrdf(f0, rough, nv);
  let r = reflect(-v, n);
  let spec = textureSampleLevel(envSpecular, envSampler, equirectUv(r), rough * u.envMaxLod).rgb * brdf;
  let diffuse = irradiance(n) * base.rgb * (1.0 - metal) * (1.0 - brdf);
  return tonemap((diffuse + spec) * u.exposure);
}

@fragment
fn fs(in: VSOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var color: vec3f;
  switch u.viewMode {
    case 0u: { color = shadeFlat(in, idColor(in.ids.x * 7919u + 1u)); }
    case 1u: { color = shadeFlat(in, idColor(in.ids.x * 131071u + in.ids.y * 7u + 3u)); }
    case 3u: { color = shadeTextured(in, front); }
    default: { color = shadeFlat(in, vec3f(0.75, 0.74, 0.72)); }
  }
  return vec4f(pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)), 1.0);
}

struct SkyOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn skyVs(@builtin(vertex_index) i: u32) -> SkyOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0; // fullscreen triangle
  var o: SkyOut;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}

@fragment
fn skyFs(in: SkyOut) -> @location(0) vec4f {
  let h = u.invViewProj * vec4f(in.ndc, 1.0, 1.0); // a point on the near plane
  let d = normalize(h.xyz / h.w - u.camPos.xyz);
  let c = tonemap(textureSampleLevel(envSky, envSampler, equirectUv(d), 0.0).rgb * u.exposure);
  return vec4f(pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)), 1.0);
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

export const prefilterShader = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: vec4f; // roughness, source texel solid angle, source mips

const PI = 3.14159265;
const SAMPLES = 256u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

fn uvDir(uv: vec2f) -> vec3f {
  let theta = uv.y * PI;
  let phi = (uv.x - 0.5) * 2.0 * PI;
  return vec3f(cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta));
}

fn equirectUv(d: vec3f) -> vec2f {
  return vec2f(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

fn hammersley(i: u32) -> vec2f {
  return vec2f(f32(i) / f32(SAMPLES), f32(reverseBits(i)) * 2.3283064365386963e-10);
}

// GGX importance sampling with view = normal (the split-sum assumption). Each sample reads a source
// mip sized to its solid angle ("filtered importance sampling"), which keeps bright spots smooth.
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let n = uvDir(in.uv);
  let rough = params.x;
  if (rough <= 0.0) {
    return vec4f(textureSampleLevel(src, samp, in.uv, 0.0).rgb, 1.0);
  }
  let a2 = pow(rough, 4.0);
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.999);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var i = 0u; i < SAMPLES; i++) {
    let xi = hammersley(i);
    let phi = 2.0 * PI * xi.x;
    let cosT = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    let sinT = sqrt(1.0 - cosT * cosT);
    let h = t * (cos(phi) * sinT) + b * (sin(phi) * sinT) + n * cosT;
    let l = 2.0 * cosT * h - n;
    let nl = dot(n, l);
    if (nl > 0.0) {
      let dd = cosT * cosT * (a2 - 1.0) + 1.0;
      let pdf = a2 / (PI * dd * dd) / 4.0;
      let mip = clamp(0.5 * log2(1.0 / (f32(SAMPLES) * pdf * params.y)) + 1.0, 0.0, params.z - 1.0);
      sum += textureSampleLevel(src, samp, equirectUv(l), mip).rgb * nl;
      weight += nl;
    }
  }
  return vec4f(sum / max(weight, 1e-4), 1.0);
}
`;
