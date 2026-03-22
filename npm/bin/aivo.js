#!/usr/bin/env node

const { spawn } = require("node:child_process");
const os = require("node:os");
const { getInstalledBinaryPath, getPackageRoot } = require("../lib/paths");
const { formatLaunchError } = require("../lib/launcher");
const { shouldDelegateWindowsNpmUpdate, spawnWindowsNpmUpdate } = require("../lib/update");

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

if (shouldDelegateWindowsNpmUpdate(args, { packageRoot: getPackageRoot() })) {
  const child = spawnWindowsNpmUpdate(spawn);
  forwardExit(child);
  child.on("error", (error) => {
    console.error(`Failed to launch npm.cmd for update: ${error.message}`);
    process.exit(1);
  });
} else {
  const binaryPath = getInstalledBinaryPath();
  const child = spawn(binaryPath, args, {
    stdio: "inherit"
  });

  forwardExit(child);
  child.on("error", (error) => {
    console.error(formatLaunchError(error, binaryPath));
    process.exit(1);
  });
}
