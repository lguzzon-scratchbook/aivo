#!/usr/bin/env node

const { spawn } = require("node:child_process");
const os = require("node:os");
const { getInstalledBinaryPath } = require("../lib/paths");
const { formatLaunchError } = require("../lib/launcher");

const args = process.argv.slice(2);

function forwardExit(child) {
  child.on("exit", (code, signal) => {
    if (signal) {
      process.exitCode = 128 + (os.constants.signals[signal] || 1);
      return;
    }

    process.exit(code ?? 1);
  });
}

// Every subcommand — including `update` — runs through the native binary. `update`
// self-updates in place (download + atomic rename), which works even on Windows
// where `npm install -g` can't overwrite the in-use global package binary.
const binaryPath = getInstalledBinaryPath();
const child = spawn(binaryPath, args, {
  stdio: "inherit"
});

forwardExit(child);
child.on("error", (error) => {
  console.error(formatLaunchError(error, binaryPath));
  process.exit(1);
});
