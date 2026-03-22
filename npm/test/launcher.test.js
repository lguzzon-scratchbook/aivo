"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { formatLaunchError } = require("../lib/launcher");

test("formatLaunchError explains missing binaries and repair steps", () => {
  const error = Object.assign(new Error("spawn ENOENT"), { code: "ENOENT" });
  const message = formatLaunchError(error, "/tmp/aivo", "linux");

  assert.match(message, /aivo binary is not available at \/tmp\/aivo/);
  assert.match(message, /--ignore-scripts/);
  assert.match(message, /Repair with: npm install -g @yuanchuan\/aivo@latest/);
  assert.match(message, /Launch error: ENOENT/);
});

test("formatLaunchError adds the Windows terminal hint only on win32", () => {
  const windowsMessage = formatLaunchError(new Error("missing"), "C:\\aivo.exe", "win32");
  const unixMessage = formatLaunchError(new Error("missing"), "/tmp/aivo", "linux");

  assert.match(windowsMessage, /open a new terminal/i);
  assert.doesNotMatch(unixMessage, /open a new terminal/i);
});
