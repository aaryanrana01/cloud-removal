import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In development the React app runs on :5173 and forwards API calls to FastAPI on :7860.
// In production FastAPI serves the built files itself, so no proxy is needed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:7860',
      '/samples': 'http://localhost:7860',
    },
  },
})
