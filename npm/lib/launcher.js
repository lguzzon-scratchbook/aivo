"use strict";

const REPAIR_COMMAND = "npm install -g @yuanchuan/aivo@latest";

function formatLaunchError(error, binaryPath, platform = process.platform) {
  const lines = [
    `aivo binary is not available at ${binaryPath}.`,
    "This usually means the npm postinstall download did not complete.",
    "If you installed with --ignore-scripts, reinstall without it.",
    `Repair with: ${REPAIR_COMMAND}`
  ];

  if (error && error.code) {
    lines.push(`Launch error: ${error.code}`);
  }

  if (platform === "win32") {
    lines.push("If you just installed aivo, open a new terminal and try again.");
  }

  return lines.join("\n");
}

module.exports = {
  REPAIR_COMMAND,
  formatLaunchError
};
