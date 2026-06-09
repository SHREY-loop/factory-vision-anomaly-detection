import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        entryFileNames: 'assets/index.js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.names?.some((name) => name.endsWith('.css'))) {
            return 'assets/index.css';
          }
          return 'assets/[name][extname]';
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/predict': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/parts': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
