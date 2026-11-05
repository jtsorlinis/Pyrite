// Loads a .glb/.gltf with three.js' GLTFLoader (handles Draco, meshopt, quantization, node
// hierarchies, GPU instancing), then flattens every triangle mesh into one world-space
// position + index buffer. Textures and materials are stripped before parsing: we only need shape.

import { Matrix4, Vector3, type Mesh, type InstancedMesh, type BufferGeometry } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'meshoptimizer';

export interface RawMesh {
  positions: Float32Array; // xyz, world space
  indices: Uint32Array;
}

const GEOMETRY_EXTENSIONS = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
  'KHR_meshopt_compression',
  'KHR_mesh_quantization',
  'EXT_mesh_gpu_instancing',
]);

export async function loadGltf(data: ArrayBuffer): Promise<RawMesh> {
  const loader = new GLTFLoader();
  const draco = new DRACOLoader();
  draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
  loader.setDRACOLoader(draco);
  loader.setMeshoptDecoder(MeshoptDecoder);

  const gltf = await loader.parseAsync(stripMaterials(data), '');
  draco.dispose();
  gltf.scene.updateMatrixWorld(true);

  const parts: { geo: BufferGeometry; matrix: Matrix4 }[] = [];
  gltf.scene.traverse((obj) => {
    const mesh = obj as Mesh;
    if (!mesh.isMesh || !mesh.geometry?.attributes.position) return;
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
  const indices = new Uint32Array(iCount - (iCount % 3));
  const v = new Vector3();
  let vo = 0;
  let io = 0;
  for (const { geo, matrix } of parts) {
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(matrix);
      positions[(vo + i) * 3] = v.x;
      positions[(vo + i) * 3 + 1] = v.y;
      positions[(vo + i) * 3 + 2] = v.z;
    }
    const n = geo.index ? geo.index.count : pos.count;
    const usable = n - (n % 3);
    if (geo.index) {
      const idx = geo.index.array;
      for (let i = 0; i < usable; i++) indices[io++] = idx[i] + vo;
    } else {
      for (let i = 0; i < usable; i++) indices[io++] = i + vo;
    }
    vo += pos.count;
  }
  return { positions, indices: io === indices.length ? indices : indices.slice(0, io) };
}

/** Rewrites a GLB's JSON chunk without images/textures/materials/skins/animations. */
function stripMaterials(data: ArrayBuffer): ArrayBuffer {
  const dv = new DataView(data);
  const isGlb = data.byteLength > 20 && dv.getUint32(0, true) === 0x46546c67; // 'glTF'
  let json: any;
  let rest: Uint8Array | null = null;
  if (isGlb) {
    const jsonLen = dv.getUint32(12, true);
    json = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 20, jsonLen)));
    rest = new Uint8Array(data, 20 + jsonLen);
  } else {
    json = JSON.parse(new TextDecoder().decode(new Uint8Array(data)));
  }

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

  let text = JSON.stringify(json);
  if (!isGlb) return new TextEncoder().encode(text).buffer as ArrayBuffer;
  while (text.length % 4) text += ' ';
  const jsonBytes = new TextEncoder().encode(text);
  // Re-pad in bytes in case of multi-byte characters.
  const padded = new Uint8Array(Math.ceil(jsonBytes.length / 4) * 4).fill(0x20);
  padded.set(jsonBytes);
  const out = new Uint8Array(12 + 8 + padded.length + rest!.length);
  const odv = new DataView(out.buffer);
  odv.setUint32(0, 0x46546c67, true);
  odv.setUint32(4, 2, true);
  odv.setUint32(8, out.length, true);
  odv.setUint32(12, padded.length, true);
  odv.setUint32(16, 0x4e4f534a, true); // 'JSON'
  out.set(padded, 20);
  out.set(rest!, 20 + padded.length);
  return out.buffer;
}
