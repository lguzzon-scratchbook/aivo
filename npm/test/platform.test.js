"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { resolvePlatformAsset } = require("../lib/platform");

test("resolvePlatformAsset maps darwin arm64", () => {
  assert.deepEqual(resolvePlatformAsset("darwin", "arm64"), {
    assetName: "aivo-darwin-arm64",
    binaryName: "aivo"
  });
});

test("resolvePlatformAsset maps win32 x64", () => {
  assert.deepEqual(resolvePlatformAsset("win32", "x64"), {
    assetName: "aivo-windows-x64.exe",
    binaryName: "aivo.exe"
  });
});

test("resolvePlatformAsset rejects unsupported targets", () => {
  assert.throws(() => resolvePlatformAsset("linux", "ppc64"), /Unsupported platform/);
});
