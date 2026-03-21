"use strict";

const { createHash } = require("node:crypto");

function normalizeSha256(value) {
  const trimmed = value.trim().toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(trimmed)) {
    return null;
  }
  return trimmed;
}

function parseChecksumText(text, expectedName) {
  let fallbackHash = null;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const bsdMatch = line.match(/^SHA256 \((.+)\) = ([a-fA-F0-9]{64})$/);
    if (bsdMatch) {
      const [, name, hash] = bsdMatch;
      if (!expectedName || name.endsWith(expectedName)) {
        return normalizeSha256(hash);
      }
      continue;
    }

    const parts = line.split(/\s+/);
    const hash = normalizeSha256(parts[0] || "");
    if (!hash) {
      continue;
    }

    const remainder = line.slice(parts[0].length).trim().replace(/^\*\s*/, "");
    if (!remainder) {
      fallbackHash = hash;
      continue;
    }

    if (remainder === expectedName || remainder.endsWith(`/${expectedName}`)) {
      return hash;
    }
  }

  return fallbackHash;
}

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

module.exports = {
  normalizeSha256,
  parseChecksumText,
  sha256
};
