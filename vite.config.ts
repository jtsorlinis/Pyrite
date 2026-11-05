import { defineConfig, type Plugin } from 'vite';
import { readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

// Models live in public/models, so they're served as-is in dev and copied into the build.
const MODELS_DIR = 'public/models';

function findModels(root: string): string[] {
  const dir = join(root, MODELS_DIR);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => /\.(glb|gltf)$/i.test(f))
    .sort()
    .map((f) => `models/${f}`);
}

// models.json lists every model in public/models so each gets a button in the UI. The dev server
// answers it live (drop a file in and reload); the build writes it out, so static hosts like
// GitHub Pages get the same list.
function listModels(): Plugin {
  let root = '';
  return {
    name: 'list-models',
    configResolved(config) {
      root = config.root;
    },
    configureServer(server) {
      server.middlewares.use('/models.json', (_req, res) => {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(findModels(root)));
      });
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'models.json', source: JSON.stringify(findModels(root)) });
    },
  };
}

export default defineConfig({
  base: './', // relative URLs, so the build works from any path (e.g. GitHub Pages' /Pyrite/)
  plugins: [listModels()],
  worker: { format: 'es' },
  build: { target: 'es2022' },
  server: { port: 5173 },
});
