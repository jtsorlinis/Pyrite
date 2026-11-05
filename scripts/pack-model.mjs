// Packs a .glb for the web:
//   1. drops TANGENT (the renderer builds its tangent frame from screen-space derivatives),
//   2. re-encodes JPEG/PNG textures as WebP (normal maps at a higher quality: they show artefacts first),
//   3. meshopt-compresses with gltfpack at 16-bit positions/UVs and 12-bit normals. gltfpack's
//      defaults (14/12/8) are coarser: 12-bit UVs are about a texel on a 4K texture.
//
// Usage: npm run pack-model -- <in.glb> <out.glb>
// Needs cwebp on the PATH (brew install webp). gltfpack is run through npx at a pinned version.

import { readFileSync, writeFileSync, mkdtempSync, rmSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const GLTFPACK = 'gltfpack@1.3.0';
const GEOMETRY_FLAGS = ['-cc', '-vp', '16', '-vt', '16', '-vn', '12'];
const WEBP_QUALITY = { color: 92, normal: 97 };
const WEBP_FLAGS = ['-m', '6', '-sharp_yuv', '-metadata', 'none', '-quiet'];

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('Usage: npm run pack-model -- <in.glb> <out.glb>');
  process.exit(1);
}

const { json, bin } = readGlb(readFileSync(input));
if (json.buffers?.length !== 1 || json.buffers[0].uri !== undefined) {
  throw new Error('Expected a .glb with a single embedded buffer');
}
const views = json.bufferViews.map((bv) => bin.subarray(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength));

// 1. Tangents
let tangents = 0;
for (const mesh of json.meshes ?? []) {
  for (const prim of mesh.primitives ?? []) {
    if (prim.attributes.TANGENT !== undefined) tangents++;
    delete prim.attributes.TANGENT;
    for (const target of prim.targets ?? []) delete target.TANGENT;
  }
}

// 2. Textures
const sourceOf = (t) => (t === undefined ? undefined : (json.textures?.[t]?.source ?? json.textures?.[t]?.extensions?.EXT_texture_webp?.source));
const normalImages = new Set((json.materials ?? []).map((m) => sourceOf(m.normalTexture?.index)).filter((s) => s !== undefined));
const tmp = mkdtempSync(join(tmpdir(), 'pack-model-'));
const converted = new Set();
const report = [];
try {
  (json.images ?? []).forEach((image, i) => {
    const ext = { 'image/jpeg': 'jpg', 'image/png': 'png' }[image.mimeType];
    if (image.bufferView === undefined || !ext) return;
    const kind = normalImages.has(i) ? 'normal' : 'color';
    const src = join(tmp, `${i}.${ext}`);
    const dst = join(tmp, `${i}.webp`);
    writeFileSync(src, views[image.bufferView]);
    execFileSync('cwebp', ['-q', String(WEBP_QUALITY[kind]), ...WEBP_FLAGS, src, '-o', dst]);
    const webp = readFileSync(dst);
    const before = views[image.bufferView].length;
    if (webp.length < before) {
      views[image.bufferView] = webp;
      image.mimeType = 'image/webp';
      converted.add(i);
    }
    report.push(`  image ${i} (${kind}, q${WEBP_QUALITY[kind]}): ${mb(before)} -> ${mb(Math.min(webp.length, before))}`);
  });

  for (const tex of json.textures ?? []) {
    if (tex.source === undefined || !converted.has(tex.source)) continue;
    tex.extensions = { ...tex.extensions, EXT_texture_webp: { source: tex.source } };
    delete tex.source; // no JPEG fallback: that would cost the bytes we just saved
  }
  if (converted.size) {
    json.extensionsUsed = [...new Set([...(json.extensionsUsed ?? []), 'EXT_texture_webp'])];
    json.extensionsRequired = [...new Set([...(json.extensionsRequired ?? []), 'EXT_texture_webp'])];
  }

  // 3. Geometry: gltfpack drops the now-unused tangent data while it compresses.
  const mid = join(tmp, 'stage.glb');
  writeFileSync(mid, writeGlb(json, views));
  execFileSync('npx', ['-y', GLTFPACK, '-i', mid, '-o', output, ...GEOMETRY_FLAGS], { stdio: 'inherit' });
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

console.log(`${input} (${mb(statSync(input).size)}) -> ${output} (${mb(statSync(output).size)})`);
console.log(`  tangents dropped from ${tangents} primitive(s)`);
for (const line of report) console.log(line);

// --- GLB ------------------------------------------------------------------------------------------

function readGlb(buf) {
  if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error('Not a .glb file');
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString('utf8'));
  const binStart = 20 + jsonLen;
  const bin = buf.subarray(binStart + 8, binStart + 8 + buf.readUInt32LE(binStart));
  return { json, bin };
}

/** Lays the buffer views out again (4-byte aligned) around any that changed size. */
function writeGlb(json, views) {
  const parts = [];
  let offset = 0;
  views.forEach((data, i) => {
    const pad = (4 - (offset % 4)) % 4;
    if (pad) parts.push(Buffer.alloc(pad));
    offset += pad;
    json.bufferViews[i].byteOffset = offset;
    json.bufferViews[i].byteLength = data.length;
    parts.push(data);
    offset += data.length;
  });
  const binPad = (4 - (offset % 4)) % 4;
  if (binPad) parts.push(Buffer.alloc(binPad));
  json.buffers[0].byteLength = offset;
  const binChunk = Buffer.concat(parts);

  let text = Buffer.from(JSON.stringify(json), 'utf8');
  text = Buffer.concat([text, Buffer.alloc((4 - (text.length % 4)) % 4, 0x20)]);
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + text.length + 8 + binChunk.length, 8);
  const chunk = (len, type) => {
    const b = Buffer.alloc(8);
    b.writeUInt32LE(len, 0);
    b.writeUInt32LE(type, 4);
    return b;
  };
  return Buffer.concat([header, chunk(text.length, 0x4e4f534a), text, chunk(binChunk.length, 0x004e4942), binChunk]);
}

function mb(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}
