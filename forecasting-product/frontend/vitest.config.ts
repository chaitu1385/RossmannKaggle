import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["e2e/**/*.test.{ts,tsx}"],
    globals: true,
    setupFiles: ["./e2e/helpers/setup.ts"],
    testTimeout: 15000,
    css: { modules: { classNameStrategy: "non-scoped" } },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
