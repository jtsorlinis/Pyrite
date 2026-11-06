// Loads a .glb/.gltf with three.js' GLTFLoader (handles Draco, meshopt, quantization, node
// hierarchies, GPU instancing), then flattens every triangle mesh into one world-space vertex +
// index buffer. Images are stripped before three parses the file; we decode the one material's
// textures ourselves straight from the GLB so they can go to WebGPU untouched.

import { Matrix3, Matrix4, Vector3, type Mesh, type InstancedMesh, type BufferGeometry } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'meshoptimizer';

export interface RawMesh {
  positions: Float32Array; // xyz, world space
  normals: Float32Array; // xyz, world space, unit length
  uvs: Float32Array; // TEXCOORD_0, zeros when the mesh has none
  indices: Uint32Array;
  materialCount: number;
  /** Some material asks for both sides of its faces to be drawn (glTF `doubleSided`). */
  doubleSided: boolean;
  /** Decodes the model's material; null when it uses more than one (not supported). */
  material: Promise<MaterialInfo> | null;
}

export interface MaterialInfo {
  name: string;
  baseColor: ImageBitmap | null; // sRGB
  normal: ImageBitmap | null; // tangent space, glTF convention
  metallicRoughness: ImageBitmap | null; // G = roughness, B = metallic
  baseColorFactor: [number, number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  normalScale: number;
}

const MAX_TEXTURE_SIZE = 4096;

const GEOMETRY_EXTENSIONS = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
  'KHR_meshopt_compression',
  'KHR_mesh_quantization',
  'EXT_mesh_gpu_instancing',
]);

interface Gltf {
  json: any;
  isGlb: boolean;
  bin: Uint8Array | null; // GLB binary chunk
}

export async function loadGltf(data: ArrayBuffer): Promise<RawMesh> {
  const file = readGltf(data);
  const loader = new GLTFLoader();
  const draco = new DRACOLoader();
  draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
  loader.setDRACOLoader(draco);
  loader.setMeshoptDecoder(MeshoptDecoder);

  const gltf = await loader.parseAsync(stripMaterials(data, file), '');
  draco.dispose();
  gltf.scene.updateMatrixWorld(true);

  // three's parser keeps the glTF mesh/primitive each three.Mesh came from; the stripped file has
  // the same meshes, so we can look the material up in the original JSON.
  const materialOf = (obj: Mesh): number => {
    const a = gltf.parser.associations.get(obj) as { meshes?: number; primitives?: number } | undefined;
    if (a?.meshes === undefined || a.primitives === undefined) return -1;
    return file.json.meshes?.[a.meshes]?.primitives?.[a.primitives]?.material ?? -1;
  };

  const parts: { geo: BufferGeometry; matrix: Matrix4 }[] = [];
  const materials = new Set<number>();
  gltf.scene.traverse((obj) => {
    const mesh = obj as Mesh;
    if (!mesh.isMesh || !mesh.geometry?.attributes.position) return;
    materials.add(materialOf(mesh));
    const inst = obj as InstancedMesh;
    if (inst.isInstancedMesh) {
      const m = new Matrix4();
      for (let i = 0; i < inst.count; i++) {
        inst.getMatrixAt(i, m);
        parts.push({ geo: mesh.geometry, matrix: mesh.matrixWorld.clone().multiply(m) });
      }
    } else {
      parts.push({ geo: mesh.geometry, matrix: mesh.matrixWorld });
    }
  });
  if (!parts.length) throw new Error('No triangle meshes found in file');

  let vCount = 0;
  let iCount = 0;
  for (const { geo } of parts) {
    vCount += geo.attributes.position.count;
    iCount += geo.index ? geo.index.count : geo.attributes.position.count;
  }
  const positions = new Float32Array(vCount * 3);
  const normals = new Float32Array(vCount * 3);
  const uvs = new Float32Array(vCount * 2);
  const indices = new Uint32Array(iCount - (iCount % 3));
  const v = new Vector3();
  const normalMatrix = new Matrix3();
  let vo = 0;
  let io = 0;
  for (const { geo, matrix } of parts) {
    if (!geo.attributes.normal) geo.computeVertexNormals();
    const pos = geo.attributes.position;
    const nrm = geo.attributes.normal;
    const uv = geo.attributes.uv;
    normalMatrix.getNormalMatrix(matrix);
    for (let i = 0; i < pos.count; i++) {
      const o = vo + i;
      v.fromBufferAttribute(pos, i).applyMatrix4(matrix);
      positions[o * 3] = v.x;
      positions[o * 3 + 1] = v.y;
      positions[o * 3 + 2] = v.z;
      v.fromBufferAttribute(nrm, i).applyNormalMatrix(normalMatrix);
      normals[o * 3] = v.x;
      normals[o * 3 + 1] = v.y;
      normals[o * 3 + 2] = v.z;
      if (uv) {
        uvs[o * 2] = uv.getX(i);
        uvs[o * 2 + 1] = uv.getY(i);
      }
    }
    // A mirroring transform flips winding; swap two corners so front faces stay counter-clockwise.
    const flip = matrix.determinant() < 0;
    const n = geo.index ? geo.index.count : pos.count;
    const idx = geo.index?.array;
    for (let i = 0; i + 2 < n; i += 3) {
      const a = idx ? idx[i] : i;
      const b = idx ? idx[i + 1] : i + 1;
      const c = idx ? idx[i + 2] : i + 2;
      indices[io++] = a + vo;
      indices[io++] = (flip ? c : b) + vo;
      indices[io++] = (flip ? b : c) + vo;
    }
    vo += pos.count;
  }

  const only = materials.size === 1 ? [...materials][0] : null;
  if (only !== null && only >= 0) applyUvTransform(uvs, file.json.materials?.[only]);
  return {
    positions,
    normals,
    uvs,
    indices: io === indices.length ? indices : indices.slice(0, io),
    materialCount: materials.size,
    // Primitives without a material (-1) get glTF's default one, which is single-sided.
    doubleSided: [...materials].some((m) => m >= 0 && file.json.materials?.[m]?.doubleSided === true),
    material: only === null ? null : loadMaterial(file, only),
  };
}

/**
 * KHR_texture_transform, baked into the UVs because one UV set serves every map. Quantizers such as
 * gltfpack rely on it to map 16-bit UVs back to their real range, so ignoring it shifts the texture.
 */
function applyUvTransform(uvs: Float32Array, material: any) {
  const pbr = material?.pbrMetallicRoughness ?? {};
  const slots = [pbr.baseColorTexture, material?.normalTexture, pbr.metallicRoughnessTexture].filter(Boolean);
  if (!slots.length) return;
  const transforms = slots.map((s: any) => s.extensions?.KHR_texture_transform ?? null);
  const t = transforms[0];
  if (transforms.some((x: unknown) => JSON.stringify(x) !== JSON.stringify(t))) {
    console.warn('Texture maps use different KHR_texture_transform values; using the first one for all');
  }
  if (!t) return;
  if ((t.texCoord ?? 0) !== 0) {
    console.warn('KHR_texture_transform targets a UV set other than TEXCOORD_0; ignoring it');
    return;
  }
  const [ox, oy] = t.offset ?? [0, 0];
  const [sx, sy] = t.scale ?? [1, 1];
  const c = Math.cos(t.rotation ?? 0);
  const s = Math.sin(t.rotation ?? 0);
  // The spec's translation * rotation * scale, with rotation = mat3(c, s, 0, -s, c, 0, 0, 0, 1).
  for (let i = 0; i < uvs.length; i += 2) {
    const u = uvs[i] * sx;
    const v = uvs[i + 1] * sy;
    uvs[i] = c * u - s * v + ox;
    uvs[i + 1] = s * u + c * v + oy;
  }
}

// --- material -------------------------------------------------------------------------------------

async function loadMaterial(file: Gltf, index: number): Promise<MaterialInfo> {
  const def = index >= 0 ? (file.json.materials?.[index] ?? {}) : {};
  const pbr = def.pbrMetallicRoughness ?? {};
  const [baseColor, normal, metallicRoughness] = await Promise.all([
    loadTexture(file, pbr.baseColorTexture?.index),
    loadTexture(file, def.normalTexture?.index),
    loadTexture(file, pbr.metallicRoughnessTexture?.index),
  ]);
  return {
    name: def.name ?? (index >= 0 ? `material ${index}` : 'default'),
    baseColor,
    normal,
    metallicRoughness,
    baseColorFactor: pbr.baseColorFactor ?? [1, 1, 1, 1],
    metallicFactor: pbr.metallicFactor ?? 1,
    roughnessFactor: pbr.roughnessFactor ?? 1,
    normalScale: def.normalTexture?.scale ?? 1,
  };
}

async function loadTexture(file: Gltf, index: number | undefined): Promise<ImageBitmap | null> {
  if (index === undefined) return null;
  const { json, bin } = file;
  const tex = json.textures?.[index];
  const source = tex?.source ?? tex?.extensions?.EXT_texture_webp?.source ?? tex?.extensions?.EXT_texture_avif?.source;
  const image = json.images?.[source];
  if (!image) {
    console.warn(`Texture ${index} has no image we can decode (KTX2/Basis isn't supported)`);
    return null;
  }
  try {
    let blob: Blob;
    if (image.bufferView !== undefined && bin) {
      const bv = json.bufferViews[image.bufferView];
      const bytes = bin.subarray(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
      blob = new Blob([bytes as Uint8Array<ArrayBuffer>], { type: image.mimeType });
    } else if (typeof image.uri === 'string' && image.uri.startsWith('data:')) {
      blob = await (await fetch(image.uri)).blob();
    } else {
      console.warn(`Texture ${index} references an external file, which isn't supported`);
      return null;
    }
    // 'none' keeps the stored bytes: sRGB decoding happens in the GPU texture format, and normal
    // maps must not be colour-managed at all.
    const opts: ImageBitmapOptions = { colorSpaceConversion: 'none', premultiplyAlpha: 'none' };
    let bmp = await createImageBitmap(blob, opts);
    const scale = Math.min(1, MAX_TEXTURE_SIZE / Math.max(bmp.width, bmp.height));
    if (scale < 1) {
      const resized = await createImageBitmap(bmp, {
        ...opts,
        resizeWidth: Math.round(bmp.width * scale),
        resizeHeight: Math.round(bmp.height * scale),
        resizeQuality: 'high',
      });
      bmp.close();
      bmp = resized;
    }
    return bmp;
  } catch (e) {
    console.warn(`Couldn't decode texture ${index}:`, e);
    return null;
  }
}

// --- file handling --------------------------------------------------------------------------------

function readGltf(data: ArrayBuffer): Gltf {
  const dv = new DataView(data);
  if (data.byteLength > 20 && dv.getUint32(0, true) === 0x46546c67) {
    // 'glTF' header, then a JSON chunk and an optional BIN chunk.
    const jsonLen = dv.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 20, jsonLen)));
    const binStart = 20 + jsonLen;
    const bin =
      data.byteLength >= binStart + 8 && dv.getUint32(binStart + 4, true) === 0x004e4942 // 'BIN\0'
        ? new Uint8Array(data, binStart + 8, dv.getUint32(binStart, true))
        : null;
    return { json, isGlb: true, bin };
  }
  return { json: JSON.parse(new TextDecoder().decode(new Uint8Array(data))), isGlb: false, bin: null };
}

/** Rewrites the file without images/textures/materials/skins/animations: three only needs shape. */
function stripMaterials(data: ArrayBuffer, file: Gltf): ArrayBuffer {
  const json = structuredClone(file.json);
  delete json.images;
  delete json.textures;
  delete json.samplers;
  delete json.materials;
  delete json.animations;
  delete json.skins;
  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) delete prim.material;
  }
  for (const node of json.nodes ?? []) delete node.skin;
  if (json.extensionsRequired) {
    json.extensionsRequired = json.extensionsRequired.filter((e: string) => GEOMETRY_EXTENSIONS.has(e));
  }

  const text = new TextEncoder().encode(JSON.stringify(json));
  if (!file.isGlb) return text.buffer as ArrayBuffer;
  const jsonLen = new DataView(data).getUint32(12, true);
  const rest = new Uint8Array(data, 20 + jsonLen);
  const padded = new Uint8Array(Math.ceil(text.length / 4) * 4).fill(0x20);
  padded.set(text);
  const out = new Uint8Array(20 + padded.length + rest.length);
  const odv = new DataView(out.buffer);
  odv.setUint32(0, 0x46546c67, true);
  odv.setUint32(4, 2, true);
  odv.setUint32(8, out.length, true);
  odv.setUint32(12, padded.length, true);
  odv.setUint32(16, 0x4e4f534a, true); // 'JSON'
  out.set(padded, 20);
  out.set(rest, 20 + padded.length);
  return out.buffer;
}
