// Orbit camera: left-drag orbit, right-drag pan, wheel zoom.

import { add, cross, normalize, scale, type Vec3 } from './math';

export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  yaw = 0.7;
  pitch = 0.35;
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

  frame(target: Vec3, radius: number) {
    this.target = [...target];
    this.distance = radius * 2.6;
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
