// WebGPU renderer: GPU cluster selection (compute) + indirect vertex-pulled draw.

import { MAX_CLUSTER_TRIS, type BuildResult } from './builder';
import type { MaterialInfo } from './loader';
import { frustumPlanes, type Mat4, type Vec3 } from './math';
import { cullShader, drawShader, mipShader } from './shaders';

export interface FrameParams {
  viewProj: Mat4;
  camPos: Vec3;
  fovY: number;
  nanite: boolean; // false: draw the original mesh, no LOD
  threshold: number; // max projected error in CSS pixels while Nanite is on
  budget: number; // triangle cap while Nanite is on, 0 = none
  viewMode: number;
  gridSize: number;
  spacing: number;
}

export interface GpuStats {
  clusters: number;
  triangles: number;
  overflow: number;
  levels: Uint32Array;
  threshold: number; // pixel error actually used (3e38 when even the DAG roots exceed the cap)
  capped: boolean; // the triangle cap raised it above the quality setting
  gpuCullMs: number | null; // null when timestamp queries aren't supported
  gpuDrawMs: number | null;
}

const UNIFORM_SIZE = 256;
const STATS_SIZE = 16 + 32 * 4;
const SAMPLES = 4;
const READBACK_SIZE = 16 + STATS_SIZE + 32; // draw args + stats + 4 timestamps
const HIST_BINS = 512; // must match BINS in the cull shader
const HIST_GROUP = 256; // must match HIST_GROUP in the cull shader

interface Scene {
  buffers: GPUBuffer[];
  textures: GPUTexture[];
  material: { baseColorFactor: number[]; metallic: number; roughness: number; normalScale: number };
  cullBindGroup: GPUBindGroup;
  drawBindGroup: GPUBindGroup;
  numClusters: number;
  minDist: number;
}

export class Renderer {
  onStats: ((s: GpuStats) => void) | null = null;
  readonly capacity: number;

  private format: GPUTextureFormat;
  private cullPipeline: GPUComputePipeline;
  private histogramPipeline: GPUComputePipeline;
  private pickPipeline: GPUComputePipeline;
  private histBuffer: GPUBuffer;
  private lodBuffer: GPUBuffer;
  private drawPipeline: GPURenderPipeline;
  private cullLayout: GPUBindGroupLayout;
  private drawLayout: GPUBindGroupLayout;
  private uniformBuffer: GPUBuffer;
  private drawArgsBuffer: GPUBuffer;
  private statsBuffer: GPUBuffer;
  private visibleBuffer: GPUBuffer;
  private readbacks: { buffer: GPUBuffer; busy: boolean }[] = [];
  private colorTex: GPUTexture | null = null;
  private depthTex: GPUTexture | null = null;
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
        { binding: 6, visibility: C, buffer: { type: 'storage' } }, // chosen threshold
      ],
    });
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
      ],
    });

    const cullModule = device.createShaderModule({ code: cullShader, label: 'cull' });
    const drawModule = device.createShaderModule({ code: drawShader, label: 'draw' });
    this.mipModule = device.createShaderModule({ code: mipShader, label: 'mip' });
    for (const m of [cullModule, drawModule, this.mipModule]) {
      m.getCompilationInfo().then((info) => {
        for (const msg of info.messages) console[msg.type === 'error' ? 'error' : 'warn'](`${m.label}:${msg.lineNum}:${msg.linePos} ${msg.message}`);
      });
    }

    const cullPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.cullLayout] });
    const computePipeline = (entryPoint: string) =>
      device.createComputePipeline({ layout: cullPipelineLayout, compute: { module: cullModule, entryPoint } });
    this.histogramPipeline = computePipeline('histogram');
    this.pickPipeline = computePipeline('pick');
    this.cullPipeline = computePipeline('main');
    this.histBuffer = device.createBuffer({ size: HIST_BINS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.lodBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    this.drawPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.drawLayout] }),
      vertex: { module: drawModule, entryPoint: 'vs' },
      fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: SAMPLES },
    });

    this.uniformBuffer = device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawArgsBuffer = device.createBuffer({
      size: 16,
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
        querySet: device.createQuerySet({ type: 'timestamp', count: 4 }),
        resolve: device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      };
    }
  }

  setScene(r: BuildResult, material: MaterialInfo | null) {
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
        { binding: 6, resource: { buffer: this.lodBuffer } },
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
      ],
    });
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
      this.depthTex = this.device.createTexture({ size: [w, h], format: 'depth32float', sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
  }

  get aspect() {
    return this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
  }

  render(p: FrameParams) {
    this.ensureTargets();
    const d = this.device;
    const s = this.scene;

    const f = new Float32Array(this.uniformData);
    const u = new Uint32Array(this.uniformData);
    f.set(p.viewProj, 0);
    f.set(frustumPlanes(p.viewProj), 16);
    f.set([...p.camPos, 1], 40);
    // LOD error is measured in CSS pixels, so quality presets look the same at any devicePixelRatio.
    f[44] = Math.max(1, this.canvas.clientHeight) / (2 * Math.tan(p.fovY / 2));
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
    d.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
    d.queue.writeBuffer(this.drawArgsBuffer, 0, new Uint32Array([MAX_CLUSTER_TRIS * 3, 0, 0, 0]));
    d.queue.writeBuffer(this.statsBuffer, 0, new Uint32Array(STATS_SIZE / 4));

    const enc = d.createCommandEncoder();
    const ts = s ? this.timing : null;
    if (s) {
      enc.clearBuffer(this.histBuffer);
      const cp = enc.beginComputePass(
        ts ? { timestampWrites: { querySet: ts.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
      );
      const groups = [Math.ceil(s.numClusters / 64), p.gridSize * p.gridSize] as const;
      cp.setBindGroup(0, s.cullBindGroup);
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
    const rp = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.colorTex!.createView(),
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.075, g: 0.08, b: 0.095, a: 1 },
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTex!.createView(),
        depthClearValue: 0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
      timestampWrites: ts ? { querySet: ts.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } : undefined,
    });
    if (s) {
      rp.setPipeline(this.drawPipeline);
      rp.setBindGroup(0, s.drawBindGroup);
      rp.drawIndirect(this.drawArgsBuffer, 0);
    }
    rp.end();

    const rb = s ? this.readbacks.find((r) => !r.busy) : undefined;
    if (rb) {
      enc.copyBufferToBuffer(this.drawArgsBuffer, 0, rb.buffer, 0, 16);
      enc.copyBufferToBuffer(this.statsBuffer, 0, rb.buffer, 16, STATS_SIZE);
      if (ts) {
        enc.resolveQuerySet(ts.querySet, 0, 4, ts.resolve, 0);
        enc.copyBufferToBuffer(ts.resolve, 0, rb.buffer, 16 + STATS_SIZE, 32);
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
          const t = new BigUint64Array(bytes, 16 + STATS_SIZE, 4);
          const ms = (a: bigint, b: bigint) => (b > a ? Number(b - a) / 1e6 : 0);
          this.onStats?.({
            clusters: data[1],
            triangles: data[4],
            overflow: data[5],
            levels: data.slice(8, 40),
            threshold: new Float32Array(bytes, 16 + 8, 1)[0],
            capped: data[7] === 1,
            gpuCullMs: ts ? ms(t[0], t[1]) : null,
            gpuDrawMs: ts ? ms(t[2], t[3]) : null,
          });
        },
        () => (rb.busy = false),
      );
    }
  }
}
