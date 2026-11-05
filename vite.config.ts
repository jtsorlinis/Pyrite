import { defineConfig, type Plugin } from 'vite';
import { readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

// Dev-only endpoint listing .glb/.gltf files in the project root and ./models, so any model you
// drop into the folder shows up as a button in the UI.
function listModels(): Plugin {
  return {
    name: 'list-models',
    configureServer(server) {
      server.middlewares.use('/__models', (_req, res) => {
        const root = server.config.root;
        const found: string[] = [];
        for (const dir of ['', 'models', 'public']) {
          const abs = join(root, dir);
          if (!existsSync(abs)) continue;
          for (const f of readdirSync(abs)) {
            if (/\.(glb|gltf)$/i.test(f)) found.push(dir && dir !== 'public' ? `${dir}/${f}` : f);
          }
        }
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(found));
      });
    },
  };
}

export default defineConfig({
  plugins: [listModels()],
  worker: { format: 'es' },
  build: { target: 'es2022' },
  server: { port: 5173 },
});
