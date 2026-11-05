# Virtual Geometry (Nanite-style) in WebGPU

Loads a `.glb`, builds a Nanite-style cluster DAG in the browser, and every frame picks, per cluster and on the GPU, the coarsest geometry whose simplification error stays under ~1 pixel, optionally capped to a triangle budget. It's shown with the model's own textures, or with every triangle flat-coloured so you can watch the selection change as you move.

```bash
npm install
npm run dev        # http://localhost:5173
```

Needs a WebGPU browser (recent Chrome, Edge, or Safari). `dragon.glb` loads on startup. Any other `.glb` in the project folder (or `models/`) also gets a button, and you can drag and drop a file, use **Open .glb…**, or pass `?model=/path.glb`.

## Things to try

- **Texture / Triangle** view (keys 1–2). Texture uses the model's base colour, normal and metallic-roughness maps; Triangle gives each triangle its own colour.
- **Nanite enabled / disabled**: turn it off to draw the original mesh everywhere (frustum culling still applies), for comparison.
- **Quality** Low / Medium / High (while Nanite is on): allows 4 / 2 / 1 px of error. Drop it and whole regions collapse to coarser DAG levels.
- **Triangle cap** Off / 250K / 500K / 1M / 2M: if the quality setting would draw more than this, the GPU raises the pixel error just enough to fit (the stats then show it as *capped*). This mirrors how pixel-error LOD is usually run: constant quality, with a budget only as a ceiling.
- **Instances**: up to 576 copies, laid out on a square grid. 144 dragons is 453M source triangles, of which about 1% get drawn.

## How it works

**Build** (`src/builder.ts`, runs in a worker, ~5 s for the 3.1M-triangle dragon):

1. Weld vertices that share a position and UV, averaging their normals. Copies of a position with different UVs stay apart as a UV seam; grouping and locking below work on positions, so seams can't open cracks.
2. Split into clusters of ≤128 triangles with meshoptimizer (`buildMeshlets`).
3. Repeat until one cluster is left:
   - Group ~8 neighbouring clusters (greedy, by shared vertices).
   - Lock the vertices each group shares with other groups.
   - Simplify each group to 50% triangles (meshoptimizer, keeping UV seams and lightly penalising normal drift), then split it back into clusters. The simplifier only removes vertices, so every level reuses the original UVs and normals.
   - If a group can't shrink without crossing a UV seam (near the top of the DAG), simplify it again in permissive mode, with the texture smearing charged to its error. Without this the DAG stops converging and distant models keep thousands of triangles.

   Group boundaries move every level, so no edge stays locked forever.
4. Each cluster stores the bounds and error of the group that made it (`lod`) and of the group it was simplified into (`parent`). Error accumulates up the DAG, and parent spheres enclose child spheres, so projected error always grows towards the root.

**Runtime** (`src/shaders.ts`, `src/renderer.ts`):

- **Triangle cap → threshold** (two compute passes, only while a cap is set): a cluster is in the cut for every threshold between its own projected error and its parent's, so each visible (cluster, instance) adds its triangles to a histogram bin at its own error and subtracts them at its parent's. Summing the bins in order gives "triangles drawn at threshold t" for 512 thresholds from 1/64 px to 1024 px (2% apart); a one-thread pass finds the smallest one within the cap and uses it if it's coarser than the quality setting. Same frame, no feedback lag; a capped frame lands within about 2% under the cap. It roughly doubles selection cost (about 2 ms instead of 0.5 ms for 144 dragons), since every cluster of every instance is visited twice.
- **Cull pass**: one thread per (cluster, instance). A cluster is drawn when `projectedError(lod) <= threshold < projectedError(parent)`. A parent and its children read exactly the same stored values, and every cluster uses the same threshold, so the selected cut never overlaps or leaves holes, and no tree traversal is needed. The same pass does frustum culling and appends survivors to a visible list plus indirect draw args.
- **Draw**: a single `drawIndirect` with one instance per visible cluster and 384 vertices each. The vertex shader pulls indices, positions, normals and UVs from storage buffers, and triangles beyond a cluster's count collapse to nothing. The Triangle view shades flat facets from screen-space derivatives.
- **Texture view**: glTF metallic-roughness shading (Lambert + GGX, one sun plus sky light, ACES tone mapping). Textures are uploaded with a generated mip chain and capped at 4096 px. The normal map's tangent frame is built from screen-space derivatives, so no tangents are stored and simplification can't break them.

The builder was checked offline: simulated cuts on closed test meshes with UV seams have zero open edges, which means no cracks.

Keeping UV seams costs triangles. On the dragon at the default view, the textured DAG draws about 228k / 159k / 107k triangles at High / Medium / Low, versus 185k / 96k / 56k for a position-only build. Letting the simplifier cross seams everywhere gets close to the position-only numbers, but smears the texture visibly at Low.

## Not implemented (yet)

These parts of real Nanite are missing:

- **Occlusion culling**: no two-pass HZB.
- **Software rasterizer**: there's no compute rasterizer for pixel-sized triangles.
- **Hierarchical culling**: selection scans every cluster × instance instead of walking a BVH.
- **Streaming**: everything is resident on the GPU.
- **Cached build**: the DAG is rebuilt on every load, and multi-mesh scenes are flattened into one mesh.
- **Multiple materials**: texturing needs a model with a single material; other models fall back to the Triangle view. Emissive, occlusion, alpha and KTX2 textures are ignored.
