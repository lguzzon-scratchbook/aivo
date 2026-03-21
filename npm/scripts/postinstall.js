#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const https = require("node:https");
const { parseChecksumText, sha256 } = require("../lib/checksum");
const { getInstalledBinaryPath, getNativeDir, getPackageRoot } = require("../lib/paths");
const { resolvePlatformAsset } = require("../lib/platform");

const pkg = require(path.join(getPackageRoot(), "package.json"));

const MAX_REDIRECTS = 5;

async function main() {
  if (process.env.AIVO_SKIP_POSTINSTALL === "1") {
    return;
  }

  const { assetName } = resolvePlatformAsset();
  const version = pkg.version;
  const baseUrl =
    process.env.AIVO_INSTALL_BASE_URL ||
    `https://github.com/yuanchuan/aivo/releases/download/v${version}`;
  const checksumUrl = `${baseUrl}/${assetName}.sha256`;
  const binaryUrl = `${baseUrl}/${assetName}`;

  const [checksumText, binary] = await Promise.all([
    downloadText(checksumUrl),
    downloadBuffer(binaryUrl)
  ]);

  const expectedSha = parseChecksumText(checksumText, assetName);
  if (!expectedSha) {
    throw new Error(`Checksum asset for ${assetName} could not be parsed.`);
  }

  const actualSha = sha256(binary);
  if (actualSha !== expectedSha) {
    throw new Error(`Checksum verification failed for ${assetName}.`);
  }

  const nativeDir = getNativeDir();
  const binaryPath = getInstalledBinaryPath();
  fs.mkdirSync(nativeDir, { recursive: true });
  fs.writeFileSync(binaryPath, binary);

  if (process.platform !== "win32") {
    fs.chmodSync(binaryPath, 0o755);
  }

  console.log(`Installed aivo ${version} (${assetName})`);
}

function downloadText(url) {
  return downloadBuffer(url).then((buffer) => buffer.toString("utf8"));
}

function downloadBuffer(url, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith("https://") ? https : require("node:http");
    const request = proto.get(
      url,
      {
        headers: {
          "User-Agent": "@yuanchuan/aivo-installer"
        },
        timeout: 30_000
      },
      (response) => {
        const status = response.statusCode || 0;

        if (
          status >= 300 &&
          status < 400 &&
          response.headers.location &&
          redirectCount < MAX_REDIRECTS
        ) {
          response.resume();
          const location = response.headers.location;
          if (!location.startsWith("https://") && !location.startsWith("http://")) {
            reject(new Error(`Invalid redirect URL: ${location}`));
            return;
          }
          resolve(downloadBuffer(location, redirectCount + 1));
          return;
        }

        if (status < 200 || status >= 300) {
          reject(new Error(`Download failed: ${status} ${url}`));
          response.resume();
          return;
        }

        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => resolve(Buffer.concat(chunks)));
      }
    );

    request.on("timeout", () => {
      request.destroy();
      reject(new Error(`Download timed out: ${url}`));
    });
    request.on("error", reject);
  });
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
