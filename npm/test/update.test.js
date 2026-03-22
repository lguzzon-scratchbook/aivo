"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  WINDOWS_NPM_UPDATE_COMMAND,
  isNpmManagedPackageRoot,
  normalizePathForMatch,
  shouldDelegateWindowsNpmUpdate,
  spawnWindowsNpmUpdate
} = require("../lib/update");

test("shouldDelegateWindowsNpmUpdate only intercepts plain Windows update commands", () => {
  const npmRoot = "C:\\Users\\user\\AppData\\Roaming\\npm\\node_modules\\@yuanchuan\\aivo";

  assert.equal(shouldDelegateWindowsNpmUpdate(["update"], { platform: "win32", packageRoot: npmRoot }), true);
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["update", "--no-color"], {
      platform: "win32",
      packageRoot: npmRoot
    }),
    true
  );
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["update", "--force"], {
      platform: "win32",
      packageRoot: npmRoot
    }),
    false
  );
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["update", "--help"], {
      platform: "win32",
      packageRoot: npmRoot
    }),
    false
  );
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["chat"], {
      platform: "win32",
      packageRoot: npmRoot
    }),
    false
  );
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["update"], {
      platform: "linux",
      packageRoot: npmRoot
    }),
    false
  );
});

test("shouldDelegateWindowsNpmUpdate does not intercept non-npm package roots", () => {
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["update"], {
      platform: "win32",
      packageRoot: "C:\\Users\\user\\project\\aivo\\npm"
    }),
    false
  );
  assert.equal(
    shouldDelegateWindowsNpmUpdate(["update"], {
      platform: "win32",
      packageRoot: "C:\\Users\\user\\AppData\\Roaming\\npm"
    }),
    false
  );
});

test("spawnWindowsNpmUpdate uses cmd.exe and npm.cmd explicitly", () => {
  const calls = [];
  const spawnImpl = (...args) => {
    calls.push(args);
    return { on() {} };
  };

  spawnWindowsNpmUpdate(spawnImpl, { comspec: "C:\\Windows\\System32\\cmd.exe" });

  assert.deepEqual(calls, [
    [
      "C:\\Windows\\System32\\cmd.exe",
      ["/d", "/s", "/c", WINDOWS_NPM_UPDATE_COMMAND],
      { stdio: "inherit" }
    ]
  ]);
});

test("isNpmManagedPackageRoot recognizes global and local node_modules installs", () => {
  assert.equal(
    isNpmManagedPackageRoot("C:\\Users\\user\\AppData\\Roaming\\npm\\node_modules\\@yuanchuan\\aivo"),
    true
  );
  assert.equal(
    isNpmManagedPackageRoot("/tmp/project/node_modules/@yuanchuan/aivo"),
    true
  );
  assert.equal(isNpmManagedPackageRoot("/tmp/project/npm"), false);
});

test("normalizePathForMatch normalizes Windows separators and verbatim prefixes", () => {
  assert.equal(
    normalizePathForMatch("\\\\?\\C:\\Users\\user\\AppData\\Roaming\\npm\\node_modules\\@yuanchuan\\aivo"),
    "c:/users/user/appdata/roaming/npm/node_modules/@yuanchuan/aivo"
  );
});
