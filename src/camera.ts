// Orbit camera: left-drag orbit, right-drag pan, wheel zoom.

import { add, cross, dot, normalize, scale, sub, type Vec3 } from './math';

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
   * Returns to the starting view angle, looks at `center` (which becomes the orbit pivot), and backs
   * off just far enough that the scene fills `margin` of the view. The scene is `points` (xyz)
   * copied to each of `corners`: of an instance grid, only the corner copies can reach the edges of
   * the view. aspect is the view's width / height.
   */
  fit(points: Float32Array, corners: Vec3[], center: Vec3, aspect: number, margin: number) {
    this.yaw = HOME_YAW;
    this.pitch = HOME_PITCH;
    const { forward: F, right: R, up: U } = this.basis();
    const ty = Math.tan(this.fovY / 2) * margin;
    const tx = ty * aspect;
    // The scene's furthest extent along v, measured from center.
    const extent = (v: Vec3) => {
      let m = -Infinity;
      for (let i = 0; i < points.length; i += 3) m = Math.max(m, points[i] * v[0] + points[i + 1] * v[1] + points[i + 2] * v[2]);
      return m + Math.max(...corners.map((c) => dot(c, v))) - dot(center, v);
    };
    // From distance d, a point q (relative to center) is in view when |q.R| <= tx * (q.F + d) and
    // |q.U| <= ty * (q.F + d), i.e. d >= q.(+-R / tx - F) and d >= q.(+-U / ty - F).
    this.distance = Math.max(
      extent(sub(scale(R, 1 / tx), F)),
      extent(sub(scale(R, -1 / tx), F)),
      extent(sub(scale(U, 1 / ty), F)),
      extent(sub(scale(U, -1 / ty), F)),
      1e-6,
    );
    this.target = [...center];
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

