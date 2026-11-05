// Runs DAG construction off the main thread so the page stays responsive while building.

import { buildDag, type MeshInput } from './builder';

self.onmessage = async (e: MessageEvent<MeshInput>) => {
  try {
    const result = await buildDag(e.data, (stage, fraction) => self.postMessage({ type: 'progress', stage, fraction }));
    self.postMessage(
      { type: 'done', result },
      { transfer: [result.positions.buffer, result.normals.buffer, result.uvs.buffer, result.indices.buffer, result.clusters] },
    );
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
