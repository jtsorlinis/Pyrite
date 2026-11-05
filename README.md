# Virtual Geometry (Nanite-style) in WebGPU

Loads a `.glb`, builds a Nanite-style cluster DAG in the browser, and every frame picks, per cluster and on the GPU, the coarsest geometry whose simplification error stays under ~1 pixel. Clusters are flat-coloured so you can watch the selection change as you move.

```bash
npm install
npm run dev        # http://localhost:5173
```

Needs a WebGPU browser (recent Chrome, Edge, or Safari). `dragon.glb` loads on startup. Any other `.glb` in the project folder (or `models/`) also gets a button, and you can drag and drop a file, use **Open .glb…**, or pass `?model=/path.glb`.

## Things to try

- **Triangle / Cluster / Plain** colouring (keys 1–3).
- **Quality** Low / Medium / High: allows 4 / 2 / 1 px of error. Drop it and whole regions collapse to coarser DAG levels.
- **Instances**: up to 576 copies, laid out on a square grid. 144 dragons is 453M source triangles, of which ~870k get drawn.

## How it works

**Build** (`src/builder.ts`, runs in a worker, ~4 s for the 3.1M-triangle dragon):

1. Weld vertices by position (glTF splits them on UV/normal seams).
2. Split into clusters of ≤128 triangles with meshoptimizer (`buildMeshlets`).
3. Repeat until one cluster is left:
   - Group ~8 neighbouring clusters (greedy, by shared vertices).
   - Lock the vertices each group shares with other groups.
   - Simplify each group to 50% triangles (meshoptimizer), then split it back into clusters.

   Group boundaries move every level, so no edge stays locked forever.
4. Each cluster stores the bounds and error of the group that made it (`lod`) and of the group it was simplified into (`parent`). Error accumulates up the DAG, and parent spheres enclose child spheres, so projected error always grows towards the root.

**Runtime** (`src/shaders.ts`, `src/renderer.ts`):

- **Compute pass**: one thread per (cluster, instance). A cluster is drawn when `projectedError(lod) <= threshold < projectedError(parent)`. A parent and its children read exactly the same stored values, so the selected cut never overlaps or leaves holes, and no tree traversal is needed. The same pass does frustum culling and appends survivors to a visible list plus indirect draw args.
- **Draw**: a single `drawIndirect` with one instance per visible cluster and 384 vertices each. The vertex shader pulls indices and positions from storage buffers, and triangles beyond a cluster's count collapse to nothing. Normals come from screen-space derivatives, so the mesh needs no attributes.

The builder was checked offline: simulated cuts on closed test meshes have zero open edges, which means no cracks.

## Not implemented (yet)

These parts of real Nanite are missing:

- **Occlusion culling**: no two-pass HZB.
- **Software rasterizer**: there's no compute rasterizer for pixel-sized triangles.
- **Hierarchical culling**: selection scans every cluster × instance instead of walking a BVH.
- **Streaming**: everything is resident on the GPU.
- **Cached build**: the DAG is rebuilt on every load, and multi-mesh scenes are flattened into one mesh.
