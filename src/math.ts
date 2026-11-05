// Minimal column-major 4x4 matrix helpers.

export type Vec3 = [number, number, number];
export type Mat4 = Float32Array;

export function perspectiveReversedInfinite(fovY: number, aspect: number, near: number): Mat4 {
  // Reversed-Z, infinite far plane, WebGPU clip space (depth in [0, 1]): depth = near / -z_view.
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[11] = -1;
  m[14] = near;
  return m;
}

export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const m = new Float32Array(16);
  m[0] = x[0]; m[4] = x[1]; m[8] = x[2];
  m[1] = y[0]; m[5] = y[1]; m[9] = y[2];
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2];
  m[12] = -dot(x, eye);
  m[13] = -dot(y, eye);
  m[14] = -dot(z, eye);
  m[15] = 1;
  return m;
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

/** Left, right, bottom, top, near planes (xyz = inward normal, w = distance) from a view-projection. */
export function frustumPlanes(m: Mat4): Float32Array {
  const row = (i: number) => [m[i], m[4 + i], m[8 + i], m[12 + i]];
  const r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
  const planes = [
    r3.map((v, i) => v + r0[i]),
    r3.map((v, i) => v - r0[i]),
    r3.map((v, i) => v + r1[i]),
    r3.map((v, i) => v - r1[i]),
    r3.map((v, i) => v - r2[i]), // reversed Z: near plane is depth <= 1
  ];
  const out = new Float32Array(24);
  planes.forEach((p, i) => {
    const l = Math.hypot(p[0], p[1], p[2]) || 1;
    out.set([p[0] / l, p[1] / l, p[2] / l, p[3] / l], i * 4);
  });
  out.set([0, 0, 0, 1], 20); // unused 6th plane (infinite far), always passes
  return out;
}

export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const normalize = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
