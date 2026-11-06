// Orbit camera: left-drag orbit, right-drag pan, wheel zoom.

import { add, cross, dot, normalize, scale, type Vec3 } from './math';

// The starting view angle, which every reframe returns to: straight on from the front (+z), a
// little above so a grid of instances doesn't collapse into its front row.
const HOME_YAW = 0;
const HOME_PITCH = 0.35;

export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  yaw = HOME_YAW;
  pitch = HOME_PITCH;
  distance = 3;
  fovY = (50 * Math.PI) / 180;

  constructor(el: HTMLElement) {
    let last: { x: number; y: number } | null = null;
    let pan = false;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      last = { x: e.clientX, y: e.clientY };
      pan = e.button !== 0;
    });
    el.addEventListener('pointermove', (e) => {
      if (!last) return;
      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      if (pan) {
        const { right, up } = this.basis();
        const k = (this.distance * 2 * Math.tan(this.fovY / 2)) / el.clientHeight;
        this.target = add(this.target, add(scale(right, -dx * k), scale(up, dy * k)));
      } else {
        this.yaw -= dx * 0.005;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.005));
      }
    });
    const end = () => (last = null);
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
        this.distance *= Math.exp(delta * 0.0012);
      },
      { passive: false },
    );
  }

  /**
   * Returns to the starting view angle and moves the camera so the scene fills `margin` of the view
   * and sits centred in it. The scene is `points` (xyz) copied to each of `corners`: of an instance
   * grid, only the corner copies can reach the edges of the view. aspect is the view's width / height.
   * The orbit target lands at the scene's middle depth.
   */
  fit(points: Float32Array, corners: Vec3[], aspect: number, margin: number) {
    this.yaw = HOME_YAW;
    this.pitch = HOME_PITCH;
    const { forward: F, right: R, up: U } = this.basis();
    const support = (d: Vec3) => {
      let m = -Infinity;
      for (let i = 0; i < points.length; i += 3) m = Math.max(m, points[i] * d[0] + points[i + 1] * d[1] + points[i + 2] * d[2]);
      return m + Math.max(...corners.map((c) => dot(c, d)));
    };
    const ty = Math.tan(this.fovY / 2) * margin;
    const tx = ty * aspect;

    // Eye at R*s + F*f along one screen axis: every point needs |p.S - s| <= t * (p.F - f). The
    // two edge planes touch the scene at s - t*f = A and -s - t*f = B, which places the eye.
    const axis = (S: Vec3, t: number) => {
      const A = support(add(S, scale(F, -t)));
      const B = support(scale(add(S, scale(F, t)), -1));
      return { s: (A - B) / 2, f: -(A + B) / (2 * t) };
    };
    const h = axis(R, tx);
    const v = axis(U, ty);
    const f = Math.min(h.f, v.f); // the axis that needs the eye further back wins

    // Along the other axis the scene has room to spare, and the s above shares it out evenly in
    // world units, which on screen gives the near side more room than the far side (a grid sits
    // high). Centre on screen instead: find the points at each screen edge, then the s that puts
    // them equally far from the middle. Sliding the eye sideways keeps every depth, so this is
    // exact for those points; a second pass catches a different point taking over an edge.
    let sx = h.s;
    let sy = v.s;
    for (let pass = 0; pass < 2; pass++) {
      const e = this.screenEdges(points, corners, sx, sy, f, R, U, F);
      sx = centre(e.left, e.right);
      sy = centre(e.bottom, e.top);
    }

    const mid = (support(F) - support(scale(F, -1))) / 2;
    this.distance = Math.max(mid - f, 1e-6);
    this.target = add(add(scale(R, sx), scale(U, sy)), scale(F, mid));
  }

  /**
   * The scene points at the left, right, bottom and top of the screen for the eye R*sx + U*sy + F*f,
   * each as [offset along R (or U), depth in front of the eye].
   */
  private screenEdges(points: Float32Array, corners: Vec3[], sx: number, sy: number, f: number, R: Vec3, U: Vec3, F: Vec3) {
    const e = { left: [0, 1], right: [0, 1], bottom: [0, 1], top: [0, 1] };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const c of corners) {
      const cx = dot(c, R), cy = dot(c, U), cz = dot(c, F) - f;
      for (let i = 0; i < points.length; i += 3) {
        const px = points[i], py = points[i + 1], pz = points[i + 2];
        const lx = px * R[0] + py * R[1] + pz * R[2] + cx;
        const ly = px * U[0] + py * U[1] + pz * U[2] + cy;
        const d = px * F[0] + py * F[1] + pz * F[2] + cz;
        const x = (lx - sx) / d;
        const y = (ly - sy) / d;
        if (x < minX) (minX = x), (e.left = [lx, d]);
        if (x > maxX) (maxX = x), (e.right = [lx, d]);
        if (y < minY) (minY = y), (e.bottom = [ly, d]);
        if (y > maxY) (maxY = y), (e.top = [ly, d]);
      }
    }
    return e;
  }

  get eye(): Vec3 {
    const cp = Math.cos(this.pitch);
    const dir: Vec3 = [cp * Math.sin(this.yaw), Math.sin(this.pitch), cp * Math.cos(this.yaw)];
    return add(this.target, scale(dir, this.distance));
  }

  basis() {
    const cp = Math.cos(this.pitch);
    const forward: Vec3 = normalize([-cp * Math.sin(this.yaw), -Math.sin(this.pitch), -cp * Math.cos(this.yaw)]);
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    return { forward, right, up };
  }
}

/** The eye offset s that puts points lo = [offset, depth] and hi equally far either side of the middle. */
function centre(lo: number[], hi: number[]) {
  // (hi.offset - s) / hi.depth = -(lo.offset - s) / lo.depth
  return (hi[0] * lo[1] + lo[0] * hi[1]) / (lo[1] + hi[1]);
}
