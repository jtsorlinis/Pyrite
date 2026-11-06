// WebGPU renderer: GPU cluster selection (compute) + indirect vertex-pulled draw.
//
// Frame (occlusion culling is two-pass, as in Nanite):
//   1. cull (main): LOD cut + frustum, then a test against last frame's HZB. What passes goes to
//      the first draw; what last frame's depth hides is deferred.
//   2. draw the first list.
//   3. build the HZB from that depth, and retest the deferred list against it (post).
//   4. draw what post found visible: things that came into view this frame.
//   5. build the HZB again from the finished depth, for the next frame's step 1.
// With no model loaded, the first draw just clears the screen.

import { MAX_CLUSTER_TRIS, type BuildResult } from './builder';
import type { MaterialInfo } from './loader';
import { frustumPlanes, multiply, type Mat4, type Vec3 } from './math';
import { cullShader, drawShader, hzbShader, mipShader } from './shaders';

export interface FrameParams {
  view: Mat4;
  proj: Mat4; // from perspectiveReversedInfinite
  near: number; // the near plane proj was built with
  camPos: Vec3;
  fovY: number;
  nanite: boolean; // false: draw the original mesh, no LOD
  threshold: number; // max projected error in rendered pixels while Nanite is on
  budget: number; // triangle cap while Nanite is on, 0 = none
  viewMode: number;
  gridSize: number;
  spacing: number;
}

export interface GpuStats {
  clusters: number; // drawn, both draws
  lateClusters: number; // drawn by the second draw: hidden last frame, visible now
  occluded: number; // clusters occlusion culling removed
  triangles: number;
  overflow: number;
  levels: Uint32Array;
  threshold: number; // pixel error actually used (3e38 when even the DAG roots exceed the cap)
  capped: boolean; // the triangle cap raised it above the quality setting
  gpuCullMs: number | null; // selection, culling and HZB builds; null without timestamp queries
  gpuDrawMs: number | null;
}

const UNIFORM_SIZE = 512;
const STATS_SIZE = 16 + 32 * 4 + 16; // ..., levels, occluded + padding
const DRAW_ARGS_SIZE = 32; // first draw, second draw
const SAMPLES = 4;
// Begin/end timestamps of: cull, first draw, HZB + post cull, second draw, HZB.
const QUERIES = 10;
const READBACK_SIZE = DRAW_ARGS_SIZE + STATS_SIZE + QUERIES * 8;
const HIST_BINS = 512; // must match BINS in the cull shader
const HIST_GROUP = 256; // must match HIST_GROUP in the cull shader

interface Hzb {
  texture: GPUTexture;
  steps: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; groups: readonly [number, number] }[];
  mainBindGroup: GPUBindGroup; // the whole pyramid + post's dispatch args, for main
  postBindGroup: GPUBindGroup; // the whole pyramid, for post
}

interface Scene {
  buffers: GPUBuffer[];
  textures: GPUTexture[];
  material: { baseColorFactor: number[]; metallic: number; roughness: number; normalScale: number };
  cullBindGroup: GPUBindGroup;
  drawBindGroup: GPUBindGroup;
  numClusters: number;
  minDist: number;
  cullBack: boolean; // skip back-facing triangles: the material isn't double-sided
}

export class Renderer {
  onStats: ((s: GpuStats) => void) | null = null;
  readonly capacity: number;

  private format: GPUTextureFormat;
  private cullPipeline: GPUComputePipeline;
  private postPipeline: GPUComputePipeline;
  private histogramPipeline: GPUComputePipeline;
  private pickPipeline: GPUComputePipeline;
  private hzbFromDepthPipeline: GPUComputePipeline;
  private hzbDownPipeline: GPUComputePipeline;
  private histBuffer: GPUBuffer;
  private cullStateBuffer: GPUBuffer;
  private postArgsBuffer: GPUBuffer;
  private deferredBuffer: GPUBuffer;
  /** [first draw, second draw] x [no culling, back-face culling]. */
  private drawPipelines: GPURenderPipeline[][];
  private cullLayout: GPUBindGroupLayout;
  private hzbMainLayout: GPUBindGroupLayout;
  private hzbPostLayout: GPUBindGroupLayout;
  private drawLayout: GPUBindGroupLayout;
  private uniformBuffer: GPUBuffer;
  private drawArgsBuffer: GPUBuffer;
  private statsBuffer: GPUBuffer;
  private visibleBuffer: GPUBuffer;
  private readbacks: { buffer: GPUBuffer; busy: boolean }[] = [];
  private colorTex: GPUTexture | null = null;
  private depthTex: GPUTexture | null = null;
  private hzb: Hzb | null = null;
  /** 1x1 targets for the HZB dispatch slots past the pyramid's last level. */
  private hzbStandIns: GPUTexture[];
  /** The HZB holds the last frame's finished depth, seen from `prev`. */
  private hzbValid = false;
  private prev: { view: Mat4; proj: number[] } | null = null;
  private scene: Scene | null = null;
  private uniformData = new ArrayBuffer(UNIFORM_SIZE);
  private context: GPUCanvasContext;
  private timing: { querySet: GPUQuerySet; resolve: GPUBuffer } | null = null;
  private sampler: GPUSampler;
  private mipSampler: GPUSampler;
  private mipModule: GPUShaderModule;
  private mipPipelines = new Map<GPUTextureFormat, GPURenderPipeline>();

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    device.lost.then((info) => console.error('WebGPU device lost:', info.message));
    device.addEventListener('uncapturederror', (e) => console.error((e as GPUUncapturedErrorEvent).error.message));
    return new Renderer(device, canvas);
  }

  private constructor(
    readonly device: GPUDevice,
    private canvas: HTMLCanvasElement,
  ) {
    this.format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = canvas.getContext('webgpu')!;
    ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.context = ctx;

    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.capacity = Math.min(Math.floor(limit / 8), 1 << 22);

    const C = GPUShaderStage.COMPUTE;
    const V = GPUShaderStage.VERTEX;
    const F = GPUShaderStage.FRAGMENT;
    this.cullLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: C, buffer: { type: 'storage' } },
        { binding: 3, visibility: C, buffer: { type: 'storage' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
        { binding: 5, visibility: C, buffer: { type: 'storage' } }, // threshold histogram
        { binding: 6, visibility: C, buffer: { type: 'storage' } }, // threshold, post's dispatch
        { binding: 7, visibility: C, buffer: { type: 'storage' } }, // deferred list
      ],
    });
    // Group 1: the HZB, plus (for main only) post's indirect dispatch args.
    const hzbEntry: GPUBindGroupLayoutEntry = { binding: 0, visibility: C, texture: { sampleType: 'unfilterable-float' } };
    this.hzbMainLayout = device.createBindGroupLayout({
      entries: [hzbEntry, { binding: 1, visibility: C, buffer: { type: 'storage' } }],
    });
    this.hzbPostLayout = device.createBindGroupLayout({ entries: [hzbEntry] });
    this.drawLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } }, // clusters
        { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } }, // visible list
        { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } }, // indices
        { binding: 4, visibility: V, buffer: { type: 'read-only-storage' } }, // positions
        { binding: 5, visibility: V, buffer: { type: 'read-only-storage' } }, // normals
        { binding: 6, visibility: V, buffer: { type: 'read-only-storage' } }, // uvs
        { binding: 7, visibility: F, sampler: { type: 'filtering' } },
        { binding: 8, visibility: F, texture: { sampleType: 'float' } }, // base colour
        { binding: 9, visibility: F, texture: { sampleType: 'float' } }, // normal map
        { binding: 10, visibility: F, texture: { sampleType: 'float' } }, // metallic-roughness
        { binding: 11, visibility: V, buffer: { type: 'read-only-storage' } }, // draw args
      ],
    });

    const cullModule = device.createShaderModule({ code: cullShader, label: 'cull' });
    const hzbModule = device.createShaderModule({ code: hzbShader, label: 'hzb' });
    const drawModule = device.createShaderModule({ code: drawShader, label: 'draw' });
    this.mipModule = device.createShaderModule({ code: mipShader, label: 'mip' });
    for (const m of [cullModule, hzbModule, drawModule, this.mipModule]) {
      m.getCompilationInfo().then((info) => {
        for (const msg of info.messages) console[msg.type === 'error' ? 'error' : 'warn'](`${m.label}:${msg.lineNum}:${msg.linePos} ${msg.message}`);
      });
    }

    const mainLayout = device.createPipelineLayout({ bindGroupLayouts: [this.cullLayout, this.hzbMainLayout] });
    const postLayout = device.createPipelineLayout({ bindGroupLayouts: [this.cullLayout, this.hzbPostLayout] });
    const computePipeline = (entryPoint: string, layout = mainLayout) =>
      device.createComputePipeline({ layout, compute: { module: cullModule, entryPoint } });
    this.histogramPipeline = computePipeline('histogram');
    this.pickPipeline = computePipeline('pick');
    this.cullPipeline = computePipeline('main');
    this.postPipeline = computePipeline('post', postLayout);
    this.hzbFromDepthPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: hzbModule, entryPoint: 'fromDepth' } });
    this.hzbDownPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: hzbModule, entryPoint: 'down' } });
    this.histBuffer = device.createBuffer({ size: HIST_BINS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.cullStateBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.postArgsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.deferredBuffer = device.createBuffer({ size: this.capacity * 8, usage: GPUBufferUsage.STORAGE });
    this.hzbStandIns = [0, 1, 2].map(() => device.createTexture({ size: [1, 1], format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING }));

    const drawPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.drawLayout] });
    const drawPipeline = (post: boolean, cullMode: GPUCullMode) =>
      device.createRenderPipeline({
        layout: drawPipelineLayout,
        vertex: { module: drawModule, entryPoint: 'vs', constants: { POST: post ? 1 : 0 } },
        fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format: this.format }] },
        // Front faces are counter-clockwise: the loader keeps glTF's winding, mirrored nodes included.
        primitive: { topology: 'triangle-list', cullMode, frontFace: 'ccw' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
        multisample: { count: SAMPLES },
      });
    this.drawPipelines = [false, true].map((post) => [drawPipeline(post, 'none'), drawPipeline(post, 'back')]);

    this.uniformBuffer = device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawArgsBuffer = device.createBuffer({
      size: DRAW_ARGS_SIZE,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.statsBuffer = device.createBuffer({
      size: STATS_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.visibleBuffer = device.createBuffer({ size: this.capacity * 8, usage: GPUBufferUsage.STORAGE });
    for (let i = 0; i < 3; i++) {
      this.readbacks.push({
        buffer: device.createBuffer({ size: READBACK_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        busy: false,
      });
    }
    this.sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      maxAnisotropy: 8,
    });
    this.mipSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    if (device.features.has('timestamp-query')) {
      this.timing = {
        querySet: device.createQuerySet({ type: 'timestamp', count: QUERIES }),
        resolve: device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      };
    }
  }

  setScene(r: BuildResult, material: MaterialInfo | null, doubleSided: boolean) {
    const d = this.device;
    if (this.scene) {
      for (const b of this.scene.buffers) b.destroy();
      for (const t of this.scene.textures) t.destroy();
      this.scene = null;
    }
    const maxBinding = d.limits.maxStorageBufferBindingSize;
    if (r.indices.byteLength > maxBinding || r.positions.byteLength > maxBinding) {
      throw new Error(
        `Mesh too large for this GPU (${(r.indices.byteLength / 2 ** 20).toFixed(0)} MB of indices, binding limit ${(maxBinding / 2 ** 20).toFixed(0)} MB).`,
      );
    }
    const upload = (data: ArrayBufferView | ArrayBuffer, label: string) => {
      const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const buf = d.createBuffer({ label, size: Math.max(16, Math.ceil(bytes.byteLength / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(buf, 0, bytes);
      return buf;
    };
    const clusterBuffer = upload(r.clusters, 'clusters');
    const indexBuffer = upload(r.indices, 'indices');
    const positionBuffer = upload(r.positions, 'positions');
    const normalBuffer = upload(r.normals, 'normals');
    const uvBuffer = upload(r.uvs, 'uvs');

    // Missing maps fall back to 1x1 textures that leave the factors unchanged.
    const baseColor = this.texture(material?.baseColor ?? null, 'rgba8unorm-srgb', [255, 255, 255, 255]);
    const normalMap = this.texture(material?.normal ?? null, 'rgba8unorm', [128, 128, 255, 255]);
    const metalRough = this.texture(material?.metallicRoughness ?? null, 'rgba8unorm', [255, 255, 255, 255]);

    const cullBindGroup = d.createBindGroup({
      layout: this.cullLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: clusterBuffer } },
        { binding: 2, resource: { buffer: this.drawArgsBuffer } },
        { binding: 3, resource: { buffer: this.visibleBuffer } },
        { binding: 4, resource: { buffer: this.statsBuffer } },
        { binding: 5, resource: { buffer: this.histBuffer } },
        { binding: 6, resource: { buffer: this.cullStateBuffer } },
        { binding: 7, resource: { buffer: this.deferredBuffer } },
      ],
    });
    const drawBindGroup = d.createBindGroup({
      layout: this.drawLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: clusterBuffer } },
        { binding: 2, resource: { buffer: this.visibleBuffer } },
        { binding: 3, resource: { buffer: indexBuffer } },
        { binding: 4, resource: { buffer: positionBuffer } },
        { binding: 5, resource: { buffer: normalBuffer } },
        { binding: 6, resource: { buffer: uvBuffer } },
        { binding: 7, resource: this.sampler },
        { binding: 8, resource: baseColor.createView() },
        { binding: 9, resource: normalMap.createView() },
        { binding: 10, resource: metalRough.createView() },
        { binding: 11, resource: { buffer: this.drawArgsBuffer } },
      ],
    });
    this.hzbValid = false; // last frame's depth is of another model
    this.scene = {
      buffers: [clusterBuffer, indexBuffer, positionBuffer, normalBuffer, uvBuffer],
      textures: [baseColor, normalMap, metalRough],
      material: {
        baseColorFactor: material?.baseColorFactor ?? [1, 1, 1, 1],
        // A mesh with no material at all gets a plain dielectric instead of glTF's default metal.
        metallic: material ? material.metallicFactor : 0,
        roughness: material ? material.roughnessFactor : 0.8,
        normalScale: material?.normalScale ?? 1,
      },
      cullBindGroup,
      drawBindGroup,
      numClusters: r.clusterCount,
      cullBack: !doubleSided,
      minDist: r.radius * 1e-6,
    };
  }

  /** Uploads an image with a full mip chain, or a 1x1 texture of `fallback` when there's no image. */
  private texture(img: ImageBitmap | null, format: GPUTextureFormat, fallback: number[]): GPUTexture {
    const d = this.device;
    if (!img) {
      const t = d.createTexture({ size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      d.queue.writeTexture({ texture: t }, new Uint8Array(fallback), { bytesPerRow: 4 }, [1, 1]);
      return t;
    }
    const levels = Math.floor(Math.log2(Math.max(img.width, img.height))) + 1;
    const t = d.createTexture({
      size: [img.width, img.height],
      format,
      mipLevelCount: levels,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    d.queue.copyExternalImageToTexture({ source: img }, { texture: t }, [img.width, img.height]);
    img.close();

    // Each level is a linear-filtered blit of the previous one (sRGB formats filter in linear space).
    let pipeline = this.mipPipelines.get(format);
    if (!pipeline) {
      pipeline = d.createRenderPipeline({
        layout: 'auto',
        vertex: { module: this.mipModule, entryPoint: 'vs' },
        fragment: { module: this.mipModule, entryPoint: 'fs', targets: [{ format }] },
      });
      this.mipPipelines.set(format, pipeline);
    }
    const enc = d.createCommandEncoder();
    for (let level = 1; level < levels; level++) {
      const bg = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: t.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: this.mipSampler },
        ],
      });
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: t.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }
    d.queue.submit([enc.finish()]);
    return t;
  }

  private ensureTargets() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.colorTex) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.colorTex?.destroy();
      this.depthTex?.destroy();
      this.colorTex = this.device.createTexture({ size: [w, h], format: this.format, sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      this.depthTex = this.device.createTexture({
        size: [w, h],
        format: 'depth32float',
        sampleCount: SAMPLES,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.createHzb(w, h);
    }
  }

  /** The depth pyramid for a w x h target (see hzbShader for its layout). */
  private createHzb(w: number, h: number) {
    const d = this.device;
    this.hzb?.texture.destroy();
    this.hzbValid = false;
    const pow2 = (n: number) => 2 ** Math.ceil(Math.log2(Math.max(1, n)));
    const w0 = pow2(Math.ceil(w / 2));
    const h0 = pow2(Math.ceil(h / 2));
    const count = Math.log2(Math.max(w0, h0)) + 1;
    const texture = d.createTexture({
      label: 'hzb',
      size: [w0, h0],
      format: 'r32float',
      mipLevelCount: count,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const level = (i: number) => texture.createView({ baseMipLevel: i, mipLevelCount: 1 });
    // Dispatch k writes levels 4k..4k+3, the first from the depth buffer, the rest from level 4k-1.
    const steps = [];
    for (let k = 0; 4 * k < count; k++) {
      const pipeline = k === 0 ? this.hzbFromDepthPipeline : this.hzbDownPipeline;
      const src = k === 0 ? { binding: 0, resource: this.depthTex!.createView() } : { binding: 1, resource: level(4 * k - 1) };
      const outs = [0, 1, 2, 3].map((j) => ({
        binding: 2 + j,
        resource: 4 * k + j < count ? level(4 * k + j) : this.hzbStandIns[j - 1].createView(),
      }));
      // One thread per texel of the level it starts from: level 0, or the level read.
      const base = k === 0 ? 0 : 4 * k - 1;
      steps.push({
        pipeline,
        bindGroup: d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [src, ...outs] }),
        groups: [Math.ceil(Math.max(1, w0 >> base) / 16), Math.ceil(Math.max(1, h0 >> base) / 16)] as const,
      });
    }
    const pyramid = { binding: 0, resource: texture.createView() };
    this.hzb = {
      texture,
      steps,
      mainBindGroup: d.createBindGroup({
        layout: this.hzbMainLayout,
        entries: [pyramid, { binding: 1, resource: { buffer: this.postArgsBuffer } }],
      }),
      postBindGroup: d.createBindGroup({ layout: this.hzbPostLayout, entries: [pyramid] }),
    };
  }

  /** Rebuilds the whole pyramid from the depth buffer as it stands. */
  private buildHzb(pass: GPUComputePassEncoder) {
    for (const step of this.hzb!.steps) {
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      pass.dispatchWorkgroups(...step.groups);
    }
  }

  get aspect() {
    return this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
  }

  render(p: FrameParams) {
    this.ensureTargets();
    const d = this.device;
    const s = this.scene;
    const proj = [p.proj[0], p.proj[5], p.near, 0];
    const prev = this.hzbValid ? this.prev : null;

    const f = new Float32Array(this.uniformData);
    const u = new Uint32Array(this.uniformData);
    const viewProj = multiply(p.proj, p.view);
    f.set(viewProj, 0);
    f.set(frustumPlanes(viewProj), 16);
    f.set([...p.camPos, 1], 40);
    // LOD error is measured in rendered pixels (the canvas's backing size, after devicePixelRatio),
    // so "1px" means one pixel of the image actually drawn.
    f[44] = Math.max(1, this.canvas.height) / (2 * Math.tan(p.fovY / 2));
    f[45] = p.threshold;
    f[46] = s?.minDist ?? 1e-6;
    f[47] = p.spacing;
    u[48] = s?.numClusters ?? 0;
    u[49] = p.gridSize;
    u[50] = p.viewMode;
    u[51] = this.capacity;
    const m = s?.material;
    f.set(m?.baseColorFactor ?? [1, 1, 1, 1], 52);
    f[56] = m?.metallic ?? 0;
    f[57] = m?.roughness ?? 1;
    f[58] = m?.normalScale ?? 1;
    u[59] = p.nanite ? 0 : 1; // fullDetail: LOD selection off
    u[60] = Math.max(0, Math.min(0xffffffff, Math.round(p.budget)));
    f.set(p.view, 64);
    f.set(prev?.view ?? p.view, 80);
    f.set(proj, 96);
    f.set(prev?.proj ?? proj, 100);
    u[104] = this.canvas.width;
    u[105] = this.canvas.height;
    u[106] = prev ? 1 : 0; // testPrev
    d.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
    const V = MAX_CLUSTER_TRIS * 3;
    d.queue.writeBuffer(this.drawArgsBuffer, 0, new Uint32Array([V, 0, 0, 0, V, 0, 0, 0]));
    d.queue.writeBuffer(this.cullStateBuffer, 0, new Uint32Array([0])); // nothing deferred yet
    d.queue.writeBuffer(this.postArgsBuffer, 0, new Uint32Array([0, 1, 1])); // post: 0 x 1 x 1 groups
    d.queue.writeBuffer(this.statsBuffer, 0, new Uint32Array(STATS_SIZE / 4));

    const enc = d.createCommandEncoder();
    const ts = s ? this.timing : null;
    const timestamps = (i: number) =>
      ts ? { querySet: ts.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } : undefined;
    if (s) {
      enc.clearBuffer(this.histBuffer);
      const cp = enc.beginComputePass({ timestampWrites: timestamps(0) });
      const groups = [Math.ceil(s.numClusters / 64), p.gridSize * p.gridSize] as const;
      cp.setBindGroup(0, s.cullBindGroup);
      cp.setBindGroup(1, this.hzb!.mainBindGroup);
      if (p.nanite) {
        // Settle this frame's threshold: the quality setting, raised if needed to fit the cap.
        if (p.budget > 0) {
          cp.setPipeline(this.histogramPipeline);
          cp.dispatchWorkgroups(Math.ceil(s.numClusters / HIST_GROUP), groups[1]);
        }
        cp.setPipeline(this.pickPipeline);
        cp.dispatchWorkgroups(1);
      }
      cp.setPipeline(this.cullPipeline);
      cp.dispatchWorkgroups(...groups);
      cp.end();
    }

    // The first draw clears; the second loads, and resolves to the screen.
    const draw = (second: boolean, last: boolean) => {
      const rp = enc.beginRenderPass({
        colorAttachments: [
          {
            view: this.colorTex!.createView(),
            resolveTarget: last ? this.context.getCurrentTexture().createView() : undefined,
            clearValue: { r: 0.075, g: 0.08, b: 0.095, a: 1 },
            loadOp: second ? 'load' : 'clear',
            storeOp: last ? 'discard' : 'store',
          },
        ],
        depthStencilAttachment: {
          view: this.depthTex!.createView(),
          depthClearValue: 0,
          depthLoadOp: second ? 'load' : 'clear',
          depthStoreOp: s ? 'store' : 'discard', // the HZB is built from it
        },
        timestampWrites: timestamps(second ? 3 : 1),
      });
      if (s) {
        rp.setPipeline(this.drawPipelines[second ? 1 : 0][s.cullBack ? 1 : 0]);
        rp.setBindGroup(0, s.drawBindGroup);
        rp.drawIndirect(this.drawArgsBuffer, second ? 16 : 0);
      }
      rp.end();
    };
    draw(false, !s);
    if (s) {
      // What last frame's depth hid, against the depth drawn so far this frame.
      const cp = enc.beginComputePass({ timestampWrites: timestamps(2) });
      this.buildHzb(cp);
      cp.setPipeline(this.postPipeline);
      cp.setBindGroup(0, s.cullBindGroup);
      cp.setBindGroup(1, this.hzb!.postBindGroup);
      cp.dispatchWorkgroupsIndirect(this.postArgsBuffer, 0);
      cp.end();
      draw(true, true);
      // The finished depth, for next frame's first test.
      const hp = enc.beginComputePass({ timestampWrites: timestamps(4) });
      this.buildHzb(hp);
      hp.end();
    }
    this.hzbValid = s !== null;
    this.prev = { view: p.view, proj };

    const rb = s ? this.readbacks.find((r) => !r.busy) : undefined;
    const queries = QUERIES;
    if (rb) {
      enc.copyBufferToBuffer(this.drawArgsBuffer, 0, rb.buffer, 0, DRAW_ARGS_SIZE);
      enc.copyBufferToBuffer(this.statsBuffer, 0, rb.buffer, DRAW_ARGS_SIZE, STATS_SIZE);
      if (ts) {
        enc.resolveQuerySet(ts.querySet, 0, queries, ts.resolve, 0);
        enc.copyBufferToBuffer(ts.resolve, 0, rb.buffer, DRAW_ARGS_SIZE + STATS_SIZE, queries * 8);
      }
    }
    d.queue.submit([enc.finish()]);

    if (rb) {
      rb.busy = true;
      rb.buffer.mapAsync(GPUMapMode.READ).then(
        () => {
          const bytes = rb.buffer.getMappedRange().slice(0);
          rb.buffer.unmap();
          rb.busy = false;
          const data = new Uint32Array(bytes);
          const st = DRAW_ARGS_SIZE / 4; // stats, in words
          const t = new BigUint64Array(bytes, DRAW_ARGS_SIZE + STATS_SIZE, queries);
          // Pass i's time; passes past the ones that ran this frame count as 0.
          const pass = (i: number) => (2 * i + 1 < queries && t[2 * i + 1] > t[2 * i] ? Number(t[2 * i + 1] - t[2 * i]) / 1e6 : 0);
          this.onStats?.({
            clusters: data[1] + data[5],
            lateClusters: data[5],
            occluded: data[st + 36],
            triangles: data[st],
            overflow: data[st + 1],
            levels: data.slice(st + 4, st + 36),
            threshold: new Float32Array(bytes, DRAW_ARGS_SIZE + 8, 1)[0],
            capped: data[st + 3] === 1,
            gpuCullMs: ts ? pass(0) + pass(2) + pass(4) : null,
            gpuDrawMs: ts ? pass(1) + pass(3) : null,
          });
        },
        () => (rb.busy = false),
      );
    }
  }
}
