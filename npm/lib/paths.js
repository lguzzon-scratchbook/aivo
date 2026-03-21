"use strict";

const path = require("node:path");
const { resolvePlatformAsset } = require("./platform");

function getPackageRoot() {
  return path.resolve(__dirname, "..");
}

function getNativeDir() {
  return path.join(getPackageRoot(), "native");
}

function getInstalledBinaryPath(platform = process.platform, arch = process.arch) {
  const { binaryName } = resolvePlatformAsset(platform, arch);
  return path.join(getNativeDir(), binaryName);
}

module.exports = {
  getInstalledBinaryPath,
  getNativeDir,
  getPackageRoot
};
