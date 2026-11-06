# Pyrite

Virtual geometry (Nanite-style) in WebGPU. Fool's Nanite, if you like.

**Experience it here: [jtsorlinis.github.io/Pyrite](https://jtsorlinis.github.io/Pyrite/)** (needs a WebGPU browser)

Loads a `.glb`, builds a Nanite-style cluster DAG in the browser, and every frame picks, per cluster and on the GPU, the coarsest geometry whose simplification error stays under ~1 pixel, optionally capped to a triangle budget. It's shown with the model's own textures, or with every triangle flat-coloured so you can watch the selection change as you move.

```bash
npm install
npm run dev        # http://localhost:5173
```

Needs a WebGPU browser (recent Chrome, Edge, or Safari). `public/models/dragon.glb` loads on startup. Any other `.glb` in `public/models/` also gets a button, and you can drag and drop a file, use **Open .glb…**, or pass `?model=path.glb`.

`npm run build` writes a static site to `dist/` that works from any path, GitHub Pages included: the models are copied in, and the list of them goes into `models.json`.

`npm run deploy` builds and pushes `dist/` to the `gh-pages` branch, which GitHub Pages serves.

The bundled models are packed for the web with `npm run pack-model -- in.glb public/models/out.glb` (needs `cwebp`: `brew install webp`), which takes the dragon from 150 MB to 32 MB and the warrior from 92 MB to 24 MB:

- **Tangents dropped**: the renderer never reads them.
- **Textures as WebP**: q92 for colour and roughness, q97 for normal maps. Against the original JPEGs that's about 41 dB PSNR for colour, and normals within 0.8° on average (2.2° at the 99th percentile).
- **Geometry quantized and meshopt-compressed** (gltfpack, 16-bit positions and UVs, 12-bit normals): positions move at most 1.4% of a median triangle edge, UVs at most 0.03 texels at 4K, and normals at most 0.06°. gltfpack's defaults are coarser (12-bit UVs are about a texel at 4K).

A close-up of the dragon renders at 40 dB PSNR against the original, with Nanite off. GitHub refuses files over 100 MB, so pack any large model you add the same way.

## Things to try

- **Texture / Triangle** view (keys 1–2). Texture uses the model's base colour, normal and metallic-roughness maps; Triangle gives each triangle its own colour.
- **Nanite enabled / disabled**: turn it off to draw the original mesh everywhere (culling still applies), for comparison.
- **Quality** Low / Medium / High (while Nanite is on): allows 4 / 2 / 1 px of error, measured in rendered pixels (so on a 2× display, High is half a CSS pixel). Drop it and whole regions collapse to coarser DAG levels.
- **Triangle cap** (5M by default, or off). If the quality setting would draw more than that, the GPU raises the pixel error just enough to fit (the stats then show it as *capped*). This mirrors how pixel-error LOD is usually run: constant quality, with a budget only as a ceiling.
- **Occlusion culling** (always on) pays off when things hide each other: drop the camera to the ground among many instances and watch *Occlusion culled* climb (with Nanite off, 169 full-detail dragons went from 114M triangles drawn to 9M, and raster time from 39 ms to 4.5 ms). On a lone model it hides little, and its fixed cost (about 0.7 ms on an Apple GPU at 1024×768, mostly the two HZB builds) outweighs that.
- **Instances**: 1, 100 or 1,000 copies, laid out on a 10×10 or 40×25 grid. Changing the count (or pressing R) returns the camera to its starting angle and fits them on screen. 100 dragons is 315M source triangles, of which about 1% get drawn.

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
- **Occlusion culling** (two-pass, as in Nanite): the cull pass also tests each cluster's bounding sphere against last frame's HZB, a depth pyramid holding the farthest depth under each texel, seen from last frame's camera. Clusters that pass are drawn; hidden ones are deferred. The HZB is then rebuilt from that first draw, a post pass retests the deferred clusters against it with this frame's camera, and the ones that show get a second draw. That's what catches anything that came out from behind something this frame. A final HZB build from the finished depth serves the next frame. The sphere test projects the exact screen rectangle of the sphere, picks the HZB level where it spans at most 2×2 texels, and culls when the sphere's nearest point is behind all four. The pyramid takes one dispatch per four levels (three at typical sizes): each reduces 16×16 tiles four levels down in workgroup memory. With the triangle cap on, the budget counts clusters in the frustum before occlusion, so the drawn count stays under it.
- **Back-face culling**: in the rasterizer (`cullMode: 'back'`), counter-clockwise front faces as glTF specifies, unless the material is double-sided. Both bundled models are marked double-sided, so they draw both sides. The loader keeps the winding through mirrored transforms, and the simplifier preserves it.
- **Draw**: two `drawIndirect`s (one per occlusion pass), with one instance per visible cluster and 384 vertices each. The vertex shader pulls indices, positions, normals and UVs from storage buffers, and triangles beyond a cluster's count collapse to nothing. The Triangle view shades flat facets from screen-space derivatives.
- **Texture view**: glTF metallic-roughness shading (Lambert + GGX, one sun plus sky light, ACES tone mapping). Textures are uploaded with a generated mip chain and capped at 4096 px. The normal map's tangent frame is built from screen-space derivatives, so no tangents are stored and simplification can't break them.

The builder was checked offline: simulated cuts on closed test meshes with UV seams have zero open edges, which means no cracks.

Keeping UV seams costs triangles. On the dragon at the default view, the textured DAG draws about 228k / 159k / 107k triangles at High / Medium / Low, versus 185k / 96k / 56k for a position-only build. Letting the simplifier cross seams everywhere gets close to the position-only numbers, but smears the texture visibly at Low.

## Not implemented (yet)

These parts of real Nanite are missing:

- **Software rasterizer**: there's no compute rasterizer for pixel-sized triangles.
- **Hierarchical culling**: selection scans every cluster × instance instead of walking a BVH.
- **Streaming**: everything is resident on the GPU.
- **Cached build**: the DAG is rebuilt on every load, and multi-mesh scenes are flattened into one mesh.
- **Multiple materials**: texturing needs a model with a single material; other models fall back to the Triangle view. Emissive, occlusion, alpha and KTX2 textures are ignored.
