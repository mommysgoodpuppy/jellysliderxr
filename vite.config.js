import { defineConfig } from "vite";
import typegpuPlugin from "unplugin-typegpu/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [typegpuPlugin(), basicSsl(),],
  build: {
    target: "esnext",
  },
});
