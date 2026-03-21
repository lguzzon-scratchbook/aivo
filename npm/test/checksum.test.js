"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { parseChecksumText, sha256 } = require("../lib/checksum");

test("parseChecksumText handles plain checksum files", () => {
  const text =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  aivo-linux-x64";
  assert.equal(
    parseChecksumText(text, "aivo-linux-x64"),
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  );
});

test("parseChecksumText handles BSD checksum format", () => {
  const text =
    "SHA256 (aivo-darwin-arm64) = 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  assert.equal(
    parseChecksumText(text, "aivo-darwin-arm64"),
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  );
});

test("sha256 hashes buffers", () => {
  assert.equal(
    sha256(Buffer.from("aivo")),
    "d771015dcce0a157e164d353a07cf3315646fd634734091c5dfd97fa9c400afc"
  );
});
