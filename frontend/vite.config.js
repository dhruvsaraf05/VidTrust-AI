import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server runs on :5173, which is the origin the backend's CORS allows.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173 },
})
