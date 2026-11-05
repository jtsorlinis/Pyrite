// Loads an equirectangular Radiance .hdr for image-based lighting. Diffuse lighting is projected to
// 9 spherical-harmonic coefficients here on the CPU; the renderer builds the blurred reflection
// maps on the GPU from the half-float pixels.

import { DataUtils, FloatType } from 'three';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';

export interface Environment {
  name: string;
  width: number;
  height: number;
  pixels: Uint16Array; // RGBA half floats, row 0 = straight up
  /** Irradiance / PI as 9 SH coefficients (xyz + pad each), so diffuse = albedo * sh(n). */
  sh: Float32Array;
}

export async function loadEnvironment(url: string): Promise<Environment> {
  const res = await fetch(url);
  if (!res.ok || (res.headers.get('content-type') ?? '').includes('text/html')) throw new Error(`${url} not found`);
  const loader = new HDRLoader();
  loader.setDataType(FloatType);
  const { width, height, data } = loader.parse(await res.arrayBuffer()) as { width: number; height: number; data: Float32Array };

  const pixels = new Uint16Array(width * height * 4);
  for (let i = 0; i < data.length; i++) pixels[i] = DataUtils.toHalfFloat(i % 4 === 3 ? 1 : data[i]);
  return { name: url.split('/').pop()!.replace(/\.hdr$/i, ''), width, height, pixels, sh: projectSH(data, width, height) };
}

/** Direction for an equirect texel; must match `equirectUv` in the shaders. */
function texelDir(x: number, y: number, w: number, h: number): [number, number, number] {
  const theta = ((y + 0.5) / h) * Math.PI; // from +Y
  const phi = ((x + 0.5) / w - 0.5) * Math.PI * 2;
  return [Math.cos(phi) * Math.sin(theta), Math.cos(theta), Math.sin(phi) * Math.sin(theta)];
}

/** Ramamoorthi & Hanrahan: radiance -> SH9, convolved with the clamped cosine lobe. */
function projectSH(data: Float32Array, w: number, h: number): Float32Array {
  const c = new Float64Array(27);
  const texelArea = ((2 * Math.PI) / w) * (Math.PI / h);
  for (let y = 0; y < h; y++) {
    const dw = texelArea * Math.sin(((y + 0.5) / h) * Math.PI);
    for (let x = 0; x < w; x++) {
      const [dx, dy, dz] = texelDir(x, y, w, h);
      const basis = shBasis(dx, dy, dz);
      const i = (y * w + x) * 4;
      for (let k = 0; k < 9; k++) {
        const b = basis[k] * dw;
        c[k * 3] += data[i] * b;
        c[k * 3 + 1] += data[i + 1] * b;
        c[k * 3 + 2] += data[i + 2] * b;
      }
    }
  }
  // Cosine-lobe convolution per band, then / PI so the shader multiplies by albedo directly.
  const band = [Math.PI, (2 * Math.PI) / 3, (2 * Math.PI) / 3, (2 * Math.PI) / 3, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4];
  const out = new Float32Array(36);
  for (let k = 0; k < 9; k++) {
    for (let ch = 0; ch < 3; ch++) out[k * 4 + ch] = (c[k * 3 + ch] * band[k]) / Math.PI;
  }
  return out;
}

function shBasis(x: number, y: number, z: number): number[] {
  return [
    0.282095,
    0.488603 * y,
    0.488603 * z,
    0.488603 * x,
    1.092548 * x * y,
    1.092548 * y * z,
    0.315392 * (3 * z * z - 1),
    1.092548 * x * z,
    0.546274 * (x * x - y * y),
  ];
}
