#!/usr/bin/env node

const { spawn } = require("node:child_process");
const os = require("node:os");
const { getInstalledBinaryPath } = require("../lib/paths");

const binaryPath = getInstalledBinaryPath();

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: "inherit"
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.exitCode = 128 + (os.constants.signals[signal] || 1);
    return;
  }

  process.exit(code ?? 1);
});

child.on("error", () => {
  console.error("aivo binary is not installed.");
  console.error("Reinstall with: npm install -g @yuanchuan/aivo");
  process.exit(1);
});
