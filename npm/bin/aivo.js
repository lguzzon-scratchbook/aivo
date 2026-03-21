#!/usr/bin/env node

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const { getInstalledBinaryPath } = require("../lib/paths");

const binaryPath = getInstalledBinaryPath();

if (!fs.existsSync(binaryPath)) {
  console.error("aivo binary is not installed.");
  console.error("Reinstall with: npm install -g @yuanchuan/aivo");
  process.exit(1);
}

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: "inherit"
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 1);
});

child.on("error", (error) => {
  console.error(`Failed to launch aivo: ${error.message}`);
  process.exit(1);
});
