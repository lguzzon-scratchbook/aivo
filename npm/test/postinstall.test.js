"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const { formatInstallError, installBinary } = require("../scripts/postinstall");

test("installBinary writes atomically and leaves only the final binary", () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "aivo-postinstall-"));
  const nativeDir = path.join(tempRoot, "native");
  const binaryPath = path.join(nativeDir, "aivo");

  installBinary({
    binary: Buffer.from("aivo"),
    binaryPath,
    nativeDir,
    platform: "linux"
  });

  assert.equal(fs.readFileSync(binaryPath, "utf8"), "aivo");
  assert.deepEqual(fs.readdirSync(nativeDir), ["aivo"]);
});

test("installBinary cleans up temp files when rename fails", () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "aivo-postinstall-"));
  const nativeDir = path.join(tempRoot, "native");
  const binaryPath = path.join(nativeDir, "aivo.exe");
  fs.mkdirSync(nativeDir, { recursive: true });
  fs.writeFileSync(binaryPath, "existing");

  const fsImpl = {
    ...fs,
    renameSync() {
      const error = new Error("EPERM: file is busy");
      error.code = "EPERM";
      throw error;
    }
  };

  assert.throws(
    () =>
      installBinary({
        binary: Buffer.from("new"),
        binaryPath,
        nativeDir,
        platform: "win32",
        fsImpl
      }),
    /Failed to install aivo\.exe/
  );

  assert.equal(fs.readFileSync(binaryPath, "utf8"), "existing");
  assert.deepEqual(fs.readdirSync(nativeDir), ["aivo.exe"]);
});

test("formatInstallError includes repair guidance and Windows note", () => {
  const message = formatInstallError(new Error("Checksum verification failed"), "win32");
  assert.match(message, /Repair with: npm install -g @yuanchuan\/aivo@latest/);
  assert.match(message, /open a new terminal/i);
});

test("getMirrorBaseUrl returns correct mirror URL", () => {
  const { getMirrorBaseUrl } = require("../scripts/postinstall");
  assert.equal(getMirrorBaseUrl("1.2.3"), "https://getaivo.dev/dl/v1.2.3");
});

test("getReleaseBaseUrl returns correct GitHub URL", () => {
  const { getReleaseBaseUrl } = require("../scripts/postinstall");
  assert.equal(getReleaseBaseUrl("1.2.3"), "https://github.com/yuanchuan/aivo/releases/download/v1.2.3");
});
