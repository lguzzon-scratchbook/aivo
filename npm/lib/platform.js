"use strict";

const PLATFORM_ASSETS = {
  darwin: {
    arm64: "aivo-darwin-arm64",
    x64: "aivo-darwin-x64"
  },
  linux: {
    arm64: "aivo-linux-arm64",
    x64: "aivo-linux-x64"
  },
  win32: {
    x64: "aivo-windows-x64.exe"
  }
};

function resolvePlatformAsset(platform = process.platform, arch = process.arch) {
  const platformAssets = PLATFORM_ASSETS[platform];
  const assetName = platformAssets && platformAssets[arch];

  if (!assetName) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}. ` +
        "Supported targets: darwin-arm64, darwin-x64, linux-arm64, linux-x64, win32-x64."
    );
  }

  return {
    assetName,
    binaryName: platform === "win32" ? "aivo.exe" : "aivo"
  };
}

module.exports = {
  PLATFORM_ASSETS,
  resolvePlatformAsset
};
