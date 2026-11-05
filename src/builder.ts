// Offline (load-time) construction of a Nanite-style cluster DAG.
//
//   1. Weld vertices by position (glTF splits vertices on UV/normal seams; we only care about shape).
//   2. Split the mesh into clusters of <= 128 triangles (level 0).
//   3. Repeat until one cluster is left (or nothing simplifies any more):
//        - partition the current clusters into groups of ~8 adjacent clusters
//        - for each group: merge its triangles, lock the vertices it shares with other groups,
//          simplify to ~50% triangles, split the result back into new clusters
//      Locking the group boundary means neighbouring groups can independently pick fine or
//      coarse geometry without cracks. Group boundaries change every level, so nothing stays
//      locked forever.
//   4. Every cluster stores (lodBounds, lodError) of the group that produced it and
//      (parentBounds, parentError) of the group it was simplified into. At runtime a cluster is
//      drawn iff  project(lod) <= threshold < project(parent). Errors and bounds only grow towards
//      the root, so this test selects exactly one consistent cut of the DAG, per cluster, in
//      parallel, with no traversal.

import { MeshoptClusterizer, MeshoptSimplifier } from 'meshoptimizer';

export const MAX_CLUSTER_TRIS = 128;
const MAX_CLUSTER_VERTS = 128;
const GROUP_SIZE = 8;
const MAX_LEVELS = 64;
const MIN_REDUCTION = 0.85; // a group must shrink to <= 85% of its triangles to count as simplified

/** Each cluster is packed as 20 x 32-bit words (80 bytes), matching `struct Cluster` in the shaders. */
export const CLUSTER_WORDS = 20;
export const ROOT_ERROR = 3.0e38;
export const NONE = 0xffffffff;

export interface LevelInfo {
  clusters: number;
  triangles: number;
}

export interface BuildResult {
  positions: Float32Array; // welded vertices, xyz
  indices: Uint32Array; // triangles of every cluster at every level, global vertex indices
  clusters: ArrayBuffer; // packed cluster records, CLUSTER_WORDS words each
  clusterCount: number;
  levels: LevelInfo[]; // clusters/triangles created at each level
  sourceTriangles: number;
  sourceVertices: number;
  center: [number, number, number];
  radius: number;
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  buildMs: number;
}

export type ProgressFn = (stage: string, fraction: number) => void;

interface Sphere {
  x: number;
  y: number;
  z: number;
  r: number;
}

interface BCluster {
  tris: Uint32Array;
  bounds: Sphere; // tight-ish bounds of this cluster's own triangles (frustum culling)
  lod: Sphere; // bounds of the group that produced this cluster
  lodError: number;
  parent: Sphere | null; // bounds of the group this cluster was simplified into
  parentError: number;
  level: number;
  parentLevel: number;
  parentGroup: number;
}

/** Reusable per-vertex scratch arrays so we never allocate O(vertexCount) per group. */
class Scratch {
  stamp: Int32Array;
  stampId = 0;
  local: Uint32Array;
  l2g: Uint32Array;
  locked: Uint8Array;
  vStart: Uint32Array;
  vFill: Uint32Array;
  constructor(vertexCount: number) {
    this.stamp = new Int32Array(vertexCount);
    this.local = new Uint32Array(vertexCount);
    this.l2g = new Uint32Array(vertexCount);
    this.locked = new Uint8Array(vertexCount);
    this.vStart = new Uint32Array(vertexCount + 1);
    this.vFill = new Uint32Array(vertexCount + 1);
  }
  next(): number {
    return ++this.stampId;
  }
}

const EMPTY_ATTRIBS = new Float32Array(0);

export async function buildDag(srcPositions: Float32Array, srcIndices: Uint32Array, progress: ProgressFn): Promise<BuildResult> {
  await Promise.all([MeshoptClusterizer.ready, MeshoptSimplifier.ready]);
  const t0 = performance.now();

  progress('Welding vertices', 0);
  const { positions, indices } = weld(srcPositions, srcIndices);
  const vertexCount = positions.length / 3;
  const sourceTriangles = indices.length / 3;
  if (sourceTriangles === 0) throw new Error('Mesh has no triangles');

  const { min, max } = aabb(positions);
  const center: [number, number, number] = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  let radius = 0;
  for (let i = 0; i < vertexCount; i++) {
    const dx = positions[i * 3] - center[0];
    const dy = positions[i * 3 + 1] - center[1];
    const dz = positions[i * 3 + 2] - center[2];
    radius = Math.max(radius, dx * dx + dy * dy + dz * dz);
  }
  radius = Math.sqrt(radius);
  const maxError = radius * 4; // effectively "no error limit": we want the triangle target to win

  progress('Building level 0 clusters', 0.02);
  const S = new Scratch(vertexCount);
  const clusters: BCluster[] = [];
  let pending: number[] = [];
  for (const tris of splitIntoClusters(indices, positions, null)) {
    const b = trisBounds(tris, positions);
    clusters.push({
      tris,
      bounds: b,
      lod: b,
      lodError: 0,
      parent: null,
      parentError: ROOT_ERROR,
      level: 0,
      parentLevel: NONE,
      parentGroup: NONE,
    });
    pending.push(clusters.length - 1);
  }

  // Rough progress model: each level processes about half the triangles of the previous one.
  const expectedWork = sourceTriangles * 2;
  let doneWork = 0;
  let groupCounter = 0;
  let level = 0;

  while (pending.length > 1 && level < MAX_LEVELS) {
    const groups = partition(pending, clusters, S);
    const locked = S.locked;
    const next: number[] = [];
    let simplifiedAny = false;

    for (let gi = 0; gi < groups.length; gi++) {
      const members = groups[gi];
      let memberTris = 0;
      for (const m of members) memberTris += clusters[m].tris.length / 3;

      const result = simplifyGroup(members, clusters, positions, locked, maxError, S);
      doneWork += memberTris;
      if ((gi & 63) === 0) {
        progress(`Level ${level + 1}: simplifying ${groups.length} groups`, 0.05 + 0.93 * Math.min(1, doneWork / expectedWork));
      }

      if (!result) {
        // Couldn't reduce this group meaningfully: carry its clusters to the next level,
        // where they'll be regrouped with different neighbours and retried.
        for (const m of members) next.push(m);
        continue;
      }
      simplifiedAny = true;

      let childError = 0;
      const childSpheres: Sphere[] = [];
      for (const m of members) {
        childError = Math.max(childError, clusters[m].lodError);
        childSpheres.push(clusters[m].lod);
      }
      // Error is accumulated so it bounds the distance to the original surface, and the group
      // sphere encloses every child's sphere -> projected error is monotonic up the DAG.
      const groupError = childError + result.error;
      const groupSphere = mergeSpheres(childSpheres);
      const groupId = groupCounter++;

      for (const m of members) {
        const c = clusters[m];
        c.parent = groupSphere;
        c.parentError = groupError;
        c.parentLevel = level + 1;
        c.parentGroup = groupId;
      }
      for (const tris of result.clusters) {
        clusters.push({
          tris,
          bounds: trisBounds(tris, positions),
          lod: groupSphere,
          lodError: groupError,
          parent: null,
          parentError: ROOT_ERROR,
          level: level + 1,
          parentLevel: NONE,
          parentGroup: NONE,
        });
        next.push(clusters.length - 1);
      }
    }

    if (!simplifiedAny) break;
    pending = next;
    level++;
  }

  progress('Packing GPU buffers', 0.99);
  return pack(clusters, positions, sourceTriangles, srcPositions.length / 3, center, radius, min, max, performance.now() - t0);
}

// ---------------------------------------------------------------------------------------------

function weld(pos: Float32Array, idx: Uint32Array): { positions: Float32Array; indices: Uint32Array } {
  const remap = MeshoptSimplifier.generatePositionRemap(pos, 3);
  const n = pos.length / 3;
  const newId = new Int32Array(n).fill(-1);
  const outIdx = new Uint32Array(idx.length);
  const outPos = new Float32Array(n * 3);
  let vc = 0;
  let ic = 0;
  const id = (v: number): number => {
    let r = newId[v];
    if (r < 0) {
      r = newId[v] = vc++;
      outPos[r * 3] = pos[v * 3];
      outPos[r * 3 + 1] = pos[v * 3 + 1];
      outPos[r * 3 + 2] = pos[v * 3 + 2];
    }
    return r;
  };
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const a = remap[idx[t]];
    const b = remap[idx[t + 1]];
    const c = remap[idx[t + 2]];
    if (a === b || b === c || a === c) continue; // degenerate after welding
    outIdx[ic++] = id(a);
    outIdx[ic++] = id(b);
    outIdx[ic++] = id(c);
  }
  return { positions: outPos.slice(0, vc * 3), indices: outIdx.slice(0, ic) };
}

function splitIntoClusters(idx: Uint32Array, pos: Float32Array, localToGlobal: Uint32Array | null): Uint32Array[] {
  const m = MeshoptClusterizer.buildMeshlets(idx, pos, 3, MAX_CLUSTER_VERTS, MAX_CLUSTER_TRIS, 0);
  const out: Uint32Array[] = [];
  for (let i = 0; i < m.meshletCount; i++) {
    const vo = m.meshlets[i * 4];
    const to = m.meshlets[i * 4 + 1];
    const tc = m.meshlets[i * 4 + 3];
    const tris = new Uint32Array(tc * 3);
    for (let k = 0; k < tc * 3; k++) {
      const v = m.vertices[vo + m.triangles[to + k]];
      tris[k] = localToGlobal ? localToGlobal[v] : v;
    }
    out.push(tris);
  }
  return out;
}

interface GroupResult {
  clusters: Uint32Array[];
  error: number;
}

function simplifyGroup(
  members: number[],
  clusters: BCluster[],
  positions: Float32Array,
  locked: Uint8Array,
  maxError: number,
  S: Scratch,
): GroupResult | null {
  // Gather the group's triangles into a compact local vertex space (meshopt's JS wrapper copies the
  // whole vertex buffer into wasm on every call, so passing the full mesh would be very slow).
  let indexCount = 0;
  for (const m of members) indexCount += clusters[m].tris.length;
  const localIdx = new Uint32Array(indexCount);
  const stamp = S.next();
  let localCount = 0;
  let k = 0;
  for (const m of members) {
    const tris = clusters[m].tris;
    for (let i = 0; i < tris.length; i++) {
      const v = tris[i];
      if (S.stamp[v] !== stamp) {
        S.stamp[v] = stamp;
        S.local[v] = localCount;
        S.l2g[localCount] = v;
        localCount++;
      }
      localIdx[k++] = S.local[v];
    }
  }
  const lpos = new Float32Array(localCount * 3);
  const lock = new Uint8Array(localCount);
  for (let l = 0; l < localCount; l++) {
    const g = S.l2g[l];
    lpos[l * 3] = positions[g * 3];
    lpos[l * 3 + 1] = positions[g * 3 + 1];
    lpos[l * 3 + 2] = positions[g * 3 + 2];
    lock[l] = locked[g];
  }

  const triCount = indexCount / 3;
  const target = Math.max(1, Math.floor(triCount / 2)) * 3;
  const [simplified, error] = MeshoptSimplifier.simplifyWithAttributes(
    localIdx,
    lpos,
    3,
    EMPTY_ATTRIBS,
    0,
    [],
    lock,
    target,
    maxError,
    ['ErrorAbsolute'],
  );
  if (simplified.length === 0 || simplified.length > indexCount * MIN_REDUCTION) return null;

  const l2g = S.l2g.slice(0, localCount);
  return { clusters: splitIntoClusters(simplified, lpos, l2g), error };
}

/**
 * Partition clusters into groups of ~GROUP_SIZE that share as many vertices as possible.
 * Greedy region growing seeded in Morton order, with a spatial fallback for disconnected pieces
 * and a final pass that folds tiny leftover groups into a neighbour. Also fills `S.locked` with
 * the vertices that are shared between different groups.
 */
function partition(ids: number[], clusters: BCluster[], S: Scratch): number[][] {
  const n = ids.length;
  S.locked.fill(0);
  if (n <= GROUP_SIZE) return [ids.slice()];

  // Unique vertex list for each cluster.
  const cvStart = new Uint32Array(n + 1);
  let upper = 0;
  for (let i = 0; i < n; i++) upper += clusters[ids[i]].tris.length;
  const cv = new Uint32Array(upper);
  let p = 0;
  for (let i = 0; i < n; i++) {
    cvStart[i] = p;
    const stamp = S.next();
    const tris = clusters[ids[i]].tris;
    for (let k = 0; k < tris.length; k++) {
      const v = tris[k];
      if (S.stamp[v] !== stamp) {
        S.stamp[v] = stamp;
        cv[p++] = v;
      }
    }
  }
  cvStart[n] = p;

  // Vertex -> clusters (CSR).
  const vStart = S.vStart;
  vStart.fill(0);
  for (let k = 0; k < p; k++) vStart[cv[k] + 1]++;
  for (let v = 1; v < vStart.length; v++) vStart[v] += vStart[v - 1];
  const vFill = S.vFill;
  vFill.set(vStart);
  const vList = new Uint32Array(p);
  for (let i = 0; i < n; i++) {
    for (let k = cvStart[i]; k < cvStart[i + 1]; k++) vList[vFill[cv[k]]++] = i;
  }

  // Cluster adjacency weighted by number of shared vertices.
  const adjStart = new Uint32Array(n + 1);
  const adjN: number[] = [];
  const adjW: number[] = [];
  const w = new Int32Array(n);
  const touched: number[] = [];
  for (let i = 0; i < n; i++) {
    adjStart[i] = adjN.length;
    for (let k = cvStart[i]; k < cvStart[i + 1]; k++) {
      const v = cv[k];
      for (let q = vStart[v]; q < vStart[v + 1]; q++) {
        const j = vList[q];
        if (j === i) continue;
        if (w[j] === 0) touched.push(j);
        w[j]++;
      }
    }
    for (const j of touched) {
      adjN.push(j);
      adjW.push(w[j]);
      w[j] = 0;
    }
    touched.length = 0;
  }
  adjStart[n] = adjN.length;

  // Morton order of cluster centres for seeding + spatial fallback.
  const cx = new Float32Array(n);
  const cy = new Float32Array(n);
  const cz = new Float32Array(n);
  const cr = new Float32Array(n);
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < n; i++) {
    const b = clusters[ids[i]].bounds;
    cx[i] = b.x; cy[i] = b.y; cz[i] = b.z; cr[i] = b.r;
    mnx = Math.min(mnx, b.x); mny = Math.min(mny, b.y); mnz = Math.min(mnz, b.z);
    mxx = Math.max(mxx, b.x); mxy = Math.max(mxy, b.y); mxz = Math.max(mxz, b.z);
  }
  const ext = Math.max(mxx - mnx, mxy - mny, mxz - mnz, 1e-20);
  const codes = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    codes[i] = morton3(((cx[i] - mnx) / ext) * 1023, ((cy[i] - mny) / ext) * 1023, ((cz[i] - mnz) / ext) * 1023);
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => codes[a] - codes[b]);
  const rank = new Uint32Array(n);
  for (let r = 0; r < n; r++) rank[order[r]] = r;

  const groupOf = new Int32Array(n).fill(-1);
  const groups: number[][] = [];
  const candW = new Float64Array(n);
  const cand: number[] = [];

  const addNeighbours = (i: number) => {
    for (let q = adjStart[i]; q < adjStart[i + 1]; q++) {
      const j = adjN[q];
      if (groupOf[j] >= 0) continue;
      if (candW[j] === 0) cand.push(j);
      candW[j] += adjW[q];
    }
  };
  const dist2 = (a: number, b: number) => {
    const dx = cx[a] - cx[b], dy = cy[a] - cy[b], dz = cz[a] - cz[b];
    return dx * dx + dy * dy + dz * dz;
  };

  for (const seed of order) {
    if (groupOf[seed] >= 0) continue;
    const g = groups.length;
    const members = [seed];
    groupOf[seed] = g;
    addNeighbours(seed);

    while (members.length < GROUP_SIZE) {
      let best = -1;
      let bestW = 0;
      let bestD = Infinity;
      for (const j of cand) {
        if (groupOf[j] >= 0) continue;
        const wj = candW[j];
        if (wj > bestW || (wj === bestW && dist2(j, seed) < bestD)) {
          best = j;
          bestW = wj;
          bestD = dist2(j, seed);
        }
      }
      if (best < 0) {
        // No connected neighbours left: try a spatially close cluster (disconnected parts).
        const r0 = rank[seed];
        for (let r = Math.max(0, r0 - 32); r < Math.min(n, r0 + 33); r++) {
          const j = order[r];
          if (groupOf[j] >= 0) continue;
          const d = dist2(j, seed);
          const lim = 3 * (cr[seed] + cr[j]);
          if (d < bestD && d < lim * lim) {
            best = j;
            bestD = d;
          }
        }
      }
      if (best < 0) break;
      groupOf[best] = g;
      members.push(best);
      addNeighbours(best);
    }
    for (const j of cand) candW[j] = 0;
    cand.length = 0;
    groups.push(members);
  }

  // Fold tiny groups into their most-connected neighbour group.
  const alive = new Uint8Array(groups.length).fill(1);
  const gw = new Map<number, number>();
  for (let g = 0; g < groups.length; g++) {
    if (!alive[g] || groups[g].length >= GROUP_SIZE / 2) continue;
    gw.clear();
    for (const i of groups[g]) {
      for (let q = adjStart[i]; q < adjStart[i + 1]; q++) {
        const h = groupOf[adjN[q]];
        if (h !== g) gw.set(h, (gw.get(h) ?? 0) + adjW[q]);
      }
    }
    let best = -1;
    let bestW = 0;
    for (const [h, wh] of gw) {
      if (alive[h] && groups[h].length + groups[g].length <= GROUP_SIZE + GROUP_SIZE / 2 && wh > bestW) {
        best = h;
        bestW = wh;
      }
    }
    if (best < 0) continue;
    for (const i of groups[g]) {
      groupOf[i] = best;
      groups[best].push(i);
    }
    groups[g] = [];
    alive[g] = 0;
  }

  // Vertices touched by clusters of more than one group are locked during simplification.
  for (let i = 0; i < n; i++) {
    const g = groupOf[i];
    for (let k = cvStart[i]; k < cvStart[i + 1]; k++) {
      const v = cv[k];
      if (S.locked[v]) continue;
      for (let q = vStart[v]; q < vStart[v + 1]; q++) {
        if (groupOf[vList[q]] !== g) {
          S.locked[v] = 1;
          break;
        }
      }
    }
  }

  const out: number[][] = [];
  for (let g = 0; g < groups.length; g++) {
    if (alive[g] && groups[g].length) out.push(groups[g].map((i) => ids[i]));
  }
  return out;
}

function morton3(x: number, y: number, z: number): number {
  const part = (v: number) => {
    let n = Math.max(0, Math.min(1023, v | 0));
    n = (n | (n << 16)) & 0x030000ff;
    n = (n | (n << 8)) & 0x0300f00f;
    n = (n | (n << 4)) & 0x030c30c3;
    n = (n | (n << 2)) & 0x09249249;
    return n;
  };
  return ((part(x) << 2) | (part(y) << 1) | part(z)) >>> 0;
}

function aabb(pos: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = pos[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return { min, max };
}

function trisBounds(tris: Uint32Array, pos: Float32Array): Sphere {
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < tris.length; i++) {
    const v = tris[i] * 3;
    const x = pos[v], y = pos[v + 1], z = pos[v + 2];
    if (x < mnx) mnx = x; if (x > mxx) mxx = x;
    if (y < mny) mny = y; if (y > mxy) mxy = y;
    if (z < mnz) mnz = z; if (z > mxz) mxz = z;
  }
  const x = (mnx + mxx) / 2, y = (mny + mxy) / 2, z = (mnz + mxz) / 2;
  let r2 = 0;
  for (let i = 0; i < tris.length; i++) {
    const v = tris[i] * 3;
    const dx = pos[v] - x, dy = pos[v + 1] - y, dz = pos[v + 2] - z;
    r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
  }
  return { x, y, z, r: Math.sqrt(r2) };
}

/** A sphere that encloses every input sphere (not minimal, but always conservative). */
function mergeSpheres(list: Sphere[]): Sphere {
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (const s of list) {
    mnx = Math.min(mnx, s.x - s.r); mxx = Math.max(mxx, s.x + s.r);
    mny = Math.min(mny, s.y - s.r); mxy = Math.max(mxy, s.y + s.r);
    mnz = Math.min(mnz, s.z - s.r); mxz = Math.max(mxz, s.z + s.r);
  }
  const x = (mnx + mxx) / 2, y = (mny + mxy) / 2, z = (mnz + mxz) / 2;
  let r = 0;
  for (const s of list) r = Math.max(r, Math.hypot(s.x - x, s.y - y, s.z - z) + s.r);
  // Tiny pad so float32 rounding on the GPU can't make a parent sphere smaller than a child's.
  return { x, y, z, r: r * (1 + 1e-6) };
}

function pack(
  clusters: BCluster[],
  positions: Float32Array,
  sourceTriangles: number,
  sourceVertices: number,
  center: [number, number, number],
  radius: number,
  aabbMin: [number, number, number],
  aabbMax: [number, number, number],
  buildMs: number,
): BuildResult {
  let totalTris = 0;
  for (const c of clusters) totalTris += c.tris.length / 3;
  const indices = new Uint32Array(totalTris * 3);
  const buf = new ArrayBuffer(clusters.length * CLUSTER_WORDS * 4);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const levels: LevelInfo[] = [];
  let off = 0;
  clusters.forEach((c, i) => {
    const b = i * CLUSTER_WORDS;
    const p = c.parent ?? c.lod;
    f[b + 0] = c.bounds.x; f[b + 1] = c.bounds.y; f[b + 2] = c.bounds.z; f[b + 3] = c.bounds.r;
    f[b + 4] = c.lod.x; f[b + 5] = c.lod.y; f[b + 6] = c.lod.z; f[b + 7] = c.lod.r;
    f[b + 8] = p.x; f[b + 9] = p.y; f[b + 10] = p.z; f[b + 11] = p.r;
    f[b + 12] = c.lodError;
    f[b + 13] = c.parentError;
    const tc = c.tris.length / 3;
    u[b + 14] = off;
    u[b + 15] = tc;
    u[b + 16] = c.level;
    u[b + 17] = c.parentLevel;
    u[b + 18] = c.parentGroup;
    u[b + 19] = 0;
    indices.set(c.tris, off * 3);
    off += tc;
    while (levels.length <= c.level) levels.push({ clusters: 0, triangles: 0 });
    levels[c.level].clusters++;
    levels[c.level].triangles += tc;
  });
  return {
    positions,
    indices,
    clusters: buf,
    clusterCount: clusters.length,
    levels,
    sourceTriangles,
    sourceVertices,
    center,
    radius,
    aabbMin,
    aabbMax,
    buildMs,
  };
}
