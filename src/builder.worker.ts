// Runs DAG construction off the main thread so the page stays responsive while building.

import { buildDag } from './builder';

export interface WorkerRequest {
  positions: Float32Array;
  indices: Uint32Array;
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  try {
    const result = await buildDag(e.data.positions, e.data.indices, (stage, fraction) =>
      self.postMessage({ type: 'progress', stage, fraction }),
    );
    self.postMessage(
      { type: 'done', result },
      { transfer: [result.positions.buffer, result.indices.buffer, result.clusters] },
    );
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
