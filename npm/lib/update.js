"use strict";

const WINDOWS_NPM_UPDATE_COMMAND = "npm.cmd install -g @yuanchuan/aivo@latest";

function normalizePathForMatch(value) {
  return String(value || "")
    .replace(/\\/g, "/")
    .replace(/^\/\/\?\//, "")
    .toLowerCase();
}

function isNpmManagedPackageRoot(packageRoot) {
  const normalized = normalizePathForMatch(packageRoot);
  return normalized.includes("/node_modules/@yuanchuan/aivo");
}

function shouldDelegateWindowsNpmUpdate(argv, options = {}) {
  const platform = options.platform || process.platform;
  const packageRoot = options.packageRoot;

  if (platform !== "win32") {
    return false;
  }

  if (!isNpmManagedPackageRoot(packageRoot)) {
    return false;
  }

  if (!Array.isArray(argv) || argv[0] !== "update") {
    return false;
  }

  const rest = argv.slice(1);
  if (rest.length === 0) {
    return true;
  }

  return rest.every((arg) => arg === "--no-color");
}

function spawnWindowsNpmUpdate(spawnImpl, options = {}) {
  const comspec = options.comspec || process.env.ComSpec || process.env.COMSPEC || "cmd.exe";
  return spawnImpl(comspec, ["/d", "/s", "/c", WINDOWS_NPM_UPDATE_COMMAND], {
    stdio: "inherit"
  });
}

module.exports = {
  WINDOWS_NPM_UPDATE_COMMAND,
  isNpmManagedPackageRoot,
  normalizePathForMatch,
  shouldDelegateWindowsNpmUpdate,
  spawnWindowsNpmUpdate
};
