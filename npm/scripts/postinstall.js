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
const REPAIR_COMMAND = "npm install -g @yuanchuan/aivo@latest";

async function main(options = {}) {
  const platform = options.platform || process.platform;
  const arch = options.arch || process.arch;
  const env = options.env || process.env;
  const fsImpl = options.fsImpl || fs;
  const logger = options.logger || console;

  if (env.AIVO_SKIP_POSTINSTALL === "1") {
    return;
  }

  const { assetName } = resolvePlatformAsset(platform, arch);
  const version = pkg.version;
  const overrideBaseUrl = env.AIVO_INSTALL_BASE_URL;

  let checksumText, binary;

  if (overrideBaseUrl) {
    [checksumText, binary] = await Promise.all([
      downloadText(`${overrideBaseUrl}/${assetName}.sha256`),
      downloadBuffer(`${overrideBaseUrl}/${assetName}`)
    ]);
  } else {
    const githubBaseUrl = getReleaseBaseUrl(version);
    const mirrorBaseUrl = getMirrorBaseUrl(version);
    [checksumText, binary] = await Promise.all([
      downloadTextWithFallback(
        `${githubBaseUrl}/${assetName}.sha256`,
        `${mirrorBaseUrl}/${assetName}.sha256`
      ),
      downloadBufferWithFallback(
        `${githubBaseUrl}/${assetName}`,
        `${mirrorBaseUrl}/${assetName}`
      )
    ]);
  }

  const expectedSha = parseChecksumText(checksumText, assetName);
  if (!expectedSha) {
    throw new Error(`Checksum asset for ${assetName} could not be parsed.`);
  }

  const actualSha = sha256(binary);
  if (actualSha !== expectedSha) {
    throw new Error(`Checksum verification failed for ${assetName}.`);
  }

  const nativeDir = getNativeDir();
  const binaryPath = getInstalledBinaryPath(platform, arch);
  installBinary({
    binary,
    binaryPath,
    nativeDir,
    platform,
    fsImpl
  });

  logger.log(`Installed aivo ${version} (${assetName})`);
  if (platform === "win32") {
    logger.log("If `aivo` is not recognized yet, open a new terminal and try again.");
  }
}

function downloadText(url) {
  return downloadBuffer(url).then((buffer) => buffer.toString("utf8"));
}

function downloadBufferWithFallback(primaryUrl, fallbackUrl) {
  return downloadBuffer(primaryUrl, 0, 10_000).catch(() => {
    return downloadBuffer(fallbackUrl);
  });
}

function downloadTextWithFallback(primaryUrl, fallbackUrl) {
  return downloadBufferWithFallback(primaryUrl, fallbackUrl).then((buffer) => buffer.toString("utf8"));
}

function getReleaseBaseUrl(version) {
  return `https://github.com/yuanchuan/aivo/releases/download/v${version}`;
}

function getMirrorBaseUrl(version) {
  return `https://getaivo.dev/dl/v${version}`;
}

function installBinary({ binary, binaryPath, nativeDir, platform, fsImpl = fs }) {
  const tempPath = path.join(
    nativeDir,
    `${path.basename(binaryPath)}.tmp-${process.pid}-${Date.now()}`
  );

  fsImpl.mkdirSync(nativeDir, { recursive: true });

  try {
    fsImpl.writeFileSync(tempPath, binary);
    if (platform !== "win32") {
      fsImpl.chmodSync(tempPath, 0o755);
    }
    fsImpl.renameSync(tempPath, binaryPath);
  } catch (error) {
    cleanupTempFile(fsImpl, tempPath);
    throw new Error(`Failed to install ${path.basename(binaryPath)}: ${error.message}`);
  }
}

function cleanupTempFile(fsImpl, tempPath) {
  if (typeof fsImpl.rmSync === "function") {
    fsImpl.rmSync(tempPath, { force: true });
    return;
  }

  try {
    fsImpl.unlinkSync(tempPath);
  } catch {
    // ignore cleanup failures
  }
}

function formatInstallError(error, platform = process.platform) {
  const lines = [error.message, `Repair with: ${REPAIR_COMMAND}`];
  if (platform === "win32") {
    lines.push("If you just installed aivo, open a new terminal and try again.");
  }
  return lines.join("\n");
}

function downloadBuffer(url, redirectCount = 0, timeout = 30_000) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith("https://") ? https : require("node:http");
    const request = proto.get(
      url,
      {
        headers: {
          "User-Agent": "@yuanchuan/aivo-installer"
        },
        timeout
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
          resolve(downloadBuffer(location, redirectCount + 1, timeout));
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

if (require.main === module) {
  main().catch((error) => {
    console.error(formatInstallError(error));
    process.exitCode = 1;
  });
}

module.exports = {
  REPAIR_COMMAND,
  downloadBuffer,
  downloadBufferWithFallback,
  downloadText,
  downloadTextWithFallback,
  formatInstallError,
  getMirrorBaseUrl,
  getReleaseBaseUrl,
  installBinary,
  main
};
