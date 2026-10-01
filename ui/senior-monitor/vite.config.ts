import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',

  server: {
    port: 8090,
    strictPort: true,
    host: '0.0.0.0',
  },

  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    pool: 'threads',
    maxWorkers: 1,
  },
});
