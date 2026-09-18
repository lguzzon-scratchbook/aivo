#!/usr/bin/env node
import { spawn, execFileSync } from 'node:child_process';
import {
  accessSync,
  constants,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  existsSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CDP_TIMEOUT_MS = 15_000;
const NAV_TIMEOUT_MS = 30_000;
const READY_POLLS = 40;
const READY_GAP_MS = 250;

const CHROME_CANDIDATES = [
  process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    : null,
  process.platform === 'darwin'
    ? '/Applications/Chromium.app/Contents/MacOS/Chromium'
    : null,
  process.platform === 'darwin'
    ? '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
    : null,
  'google-chrome',
  'google-chrome-stable',
  'chromium',
  'chromium-browser',
  'microsoft-edge',
].filter(Boolean);

function fail(message, extra = {}) {
  process.stdout.write(JSON.stringify({ ok: false, error: message, ...extra }, null, 2) + '\n');
  process.exit(2);
}

function parseArgs(argv) {
  const out = { request: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--request' || a === '-r') {
      out.request = argv[++i];
    } else if (a === '--help' || a === '-h') {
      out.help = true;
    } else if (!a.startsWith('-') && !out.request) {
      out.request = a;
    }
  }
  return out;
}

async function readRequest(path) {
  if (!path) {
    fail(
      'usage: browser-check.mjs --request <file.json|->  (JSON with url + checks[])',
    );
  }
  const raw = path === '-' ? await readStdin() : readFileSync(path, 'utf8');
  let req;
  try {
    req = JSON.parse(raw);
  } catch (e) {
    fail(`invalid JSON request: ${e.message}`);
  }
  if (!req || typeof req !== 'object' || Array.isArray(req)) {
    fail('request must be a JSON object');
  }
  if (typeof req.url !== 'string' || !req.url.trim()) {
    fail('request.url is required');
  }
  if (!Array.isArray(req.checks) || req.checks.length === 0) {
    fail('request.checks must be a non-empty array');
  }
  return req;
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let s = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (c) => (s += c));
    process.stdin.on('end', () => resolve(s));
    process.stdin.on('error', reject);
  });
}

function findChrome(explicit) {
  const list = explicit ? [explicit, ...CHROME_CANDIDATES] : CHROME_CANDIDATES;
  for (const p of list) {
    try {
      accessSync(p, constants.X_OK);
      return p;
    } catch {
    }
    if (!p.includes('/') && !p.includes('\\')) {
      try {
        const out = execFileSync(process.platform === 'win32' ? 'where' : 'which', [p], {
          encoding: 'utf8',
        })
          .trim()
          .split(/\r?\n/)[0];
        if (out) return out;
      } catch {
      }
    }
  }
  return null;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function spawnChrome(bin, profile, width, height, extraArgs = []) {
  const args = [
    '--headless=new',
    '--remote-debugging-port=0',
    `--user-data-dir=${profile}`,
    `--window-size=${width},${height}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--hide-scrollbars',
    '--disable-gpu',
    '--use-mock-keychain',
    ...extraArgs,
    'about:blank',
  ];
  return spawn(bin, args, { stdio: ['ignore', 'ignore', 'pipe'] });
}

async function waitForDevtools(profile, child) {
  const portFile = join(profile, 'DevToolsActivePort');
  for (let i = 0; i < READY_POLLS; i++) {
    if (child.exitCode != null) {
      throw new Error(`chrome exited early (code ${child.exitCode})`);
    }
    if (existsSync(portFile)) {
      const [port, path] = readFileSync(portFile, 'utf8').trim().split(/\r?\n/);
      if (port && path) {
        return { port: Number(port), browserWs: `ws://127.0.0.1:${port}${path}` };
      }
    }
    await sleep(READY_GAP_MS);
  }
  throw new Error('Chrome DevTools endpoint never came up');
}

class Cdp {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.mid = 0;
    this.pending = new Map();
    this.ws.addEventListener('message', (e) => {
      const m = JSON.parse(e.data);
      if (m.id && this.pending.has(m.id)) {
        const { res, rej, timer } = this.pending.get(m.id);
        this.pending.delete(m.id);
        clearTimeout(timer);
        m.error ? rej(new Error(JSON.stringify(m.error).slice(0, 300))) : res(m.result);
      }
    });
  }
  ready() {
    if (this.ws.readyState === WebSocket.OPEN) return Promise.resolve();
    return new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error('websocket open timed out')), CDP_TIMEOUT_MS);
      this.ws.addEventListener('open', () => {
        clearTimeout(t);
        res();
      });
      this.ws.addEventListener('error', (e) => {
        clearTimeout(t);
        rej(e);
      });
    });
  }
  send(method, params = {}, timeoutMs = CDP_TIMEOUT_MS) {
    return new Promise((res, rej) => {
      const id = ++this.mid;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        rej(new Error(`CDP timeout: ${method}`));
      }, timeoutMs);
      this.pending.set(id, { res, rej, timer });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  close() {
    try {
      this.ws.close();
    } catch {
    }
  }
}

async function openPage(browserWs) {
  const browser = new Cdp(browserWs);
  await browser.ready();
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await browser.send('Target.attachToTarget', {
    targetId,
    flatten: true,
  });
  const page = {
    send(method, params, timeoutMs) {
      return new Promise((res, rej) => {
        const id = ++browser.mid;
        const timer = setTimeout(() => {
          browser.pending.delete(id);
          rej(new Error(`CDP timeout: ${method}`));
        }, timeoutMs ?? CDP_TIMEOUT_MS);
        browser.pending.set(id, { res, rej, timer });
        browser.ws.send(
          JSON.stringify({ id, method, sessionId, params: params ?? {} }),
        );
      });
    },
    close() {
      browser.close();
    },
  };
  return page;
}

async function evaluate(page, expression) {
  const r = await page.send('Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (r.exceptionDetails) {
    throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 300));
  }
  return r.result?.value;
}

function clipFromBox(box, viewport) {
  const x = Math.max(0, box.x);
  const y = Math.max(0, box.y);
  const width = Math.max(1, Math.min(box.width, viewport[0] - x));
  const height = Math.max(1, Math.min(box.height, viewport[1] - y));
  return { x, y, width, height, scale: 1 };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    fail('usage: browser-check.mjs --request <file.json|->');
  }
  const req = await readRequest(args.request);
  const viewport = Array.isArray(req.viewport) && req.viewport.length === 2
    ? [Number(req.viewport[0]) || 1024, Number(req.viewport[1]) || 768]
    : [1024, 768];
  const outDir = req.outDir
    ? req.outDir
    : mkdtempSync(join(tmpdir(), 'aivo-browser-check-'));
  mkdirSync(outDir, { recursive: true });

  const chrome = findChrome(req.chrome);
  if (!chrome) {
    fail('no Chrome/Chromium found — pass request.chrome or install Chrome');
  }

  const profile = mkdtempSync(join(tmpdir(), 'aivo-browser-check-chrome-'));
  let child = spawnChrome(chrome, profile, viewport[0], viewport[1]);
  let page;
  const results = [];
  try {
    let devtools;
    try {
      devtools = await waitForDevtools(profile, child);
    } catch (e) {
      try {
        child.kill('SIGKILL');
      } catch {
      }
      child = spawnChrome(chrome, profile, viewport[0], viewport[1], ['--no-sandbox']);
      devtools = await waitForDevtools(profile, child);
    }
    page = await openPage(devtools.browserWs);
    await page.send('Page.enable');
    await page.send('Runtime.enable');
    await page.send('DOM.enable');
    await page.send('Emulation.setDeviceMetricsOverride', {
      width: viewport[0],
      height: viewport[1],
      deviceScaleFactor: 1,
      mobile: false,
    });
    const scheme = req.theme?.colorScheme;
    if (scheme === 'light' || scheme === 'dark' || scheme === 'no-preference') {
      await page.send('Emulation.setEmulatedMedia', {
        features: [{ name: 'prefers-color-scheme', value: scheme }],
      });
    }
    await page.send('Page.navigate', { url: req.url }, NAV_TIMEOUT_MS);
    await evaluate(
      page,
      `new Promise((r) => {
        if (document.readyState === 'complete') r(document.readyState);
        else window.addEventListener('load', () => r(document.readyState), { once: true });
        setTimeout(() => r(document.readyState), ${NAV_TIMEOUT_MS});
      })`,
    );
    await evaluate(
      page,
      `Promise.race([
        document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve(),
        new Promise((r) => setTimeout(r, 3000))
      ]).then(() => document.readyState)`,
    );
    await sleep(150);
    if (typeof req.theme?.evaluate === 'string' && req.theme.evaluate.trim()) {
      await evaluate(page, req.theme.evaluate);
      await sleep(50);
    }

    for (const check of req.checks) {
      const id = String(check.id ?? check.selector ?? 'check');
      const selector = check.selector;
      const index = Number(check.index) || 0;
      const styles = Array.isArray(check.styles) ? check.styles.map(String) : [];
      try {
        if (typeof selector !== 'string' || !selector.trim()) {
          results.push({ id, ok: false, error: 'missing selector' });
          continue;
        }
        const info = await evaluate(
          page,
          `(() => {
            const sel = ${JSON.stringify(selector)};
            const nodes = Array.from(document.querySelectorAll(sel));
            const i = ${index};
            const el = nodes[i];
            if (!el) return { count: nodes.length, missing: true };
            el.scrollIntoView({ block: 'center', inline: 'center' });
            const r = el.getBoundingClientRect();
            const cs = getComputedStyle(el);
            const wanted = ${JSON.stringify(styles)};
            const picked = {};
            for (const p of wanted) picked[p] = cs.getPropertyValue(p);
            const vis =
              r.width > 0 &&
              r.height > 0 &&
              cs.visibility !== 'hidden' &&
              cs.display !== 'none';
            return {
              count: nodes.length,
              missing: false,
              visible: vis,
              box: { x: r.x, y: r.y, width: r.width, height: r.height },
              styles: picked,
            };
          })()`,
        );
        if (info.missing) {
          results.push({
            id,
            ok: false,
            count: info.count,
            error:
              info.count === 0
                ? `no nodes match ${selector}`
                : `index ${index} out of range (${info.count} matches)`,
          });
          continue;
        }
        const row = {
          id,
          ok: true,
          count: info.count,
          visible: info.visible,
          box: info.box,
          styles: info.styles,
        };
        if (check.screenshot) {
          const png = await page.send('Page.captureScreenshot', {
            format: 'png',
            clip: clipFromBox(info.box, viewport),
            captureBeyondViewport: true,
          });
          const path = join(outDir, `${safeName(id)}.png`);
          writeFileSync(path, Buffer.from(png.data, 'base64'));
          row.screenshot = path;
        }
        if (check.hover) {
          const cx = info.box.x + info.box.width / 2;
          const cy = info.box.y + info.box.height / 2;
          await page.send('Input.dispatchMouseEvent', {
            type: 'mouseMoved',
            x: cx,
            y: cy,
          });
          await sleep(80);
          const after = await evaluate(
            page,
            `(() => {
              const el = document.querySelectorAll(${JSON.stringify(selector)})[${index}];
              if (!el) return { missing: true };
              const r = el.getBoundingClientRect();
              const cs = getComputedStyle(el);
              const wanted = ${JSON.stringify(styles)};
              const picked = {};
              for (const p of wanted) picked[p] = cs.getPropertyValue(p);
              return {
                missing: false,
                box: { x: r.x, y: r.y, width: r.width, height: r.height },
                styles: picked,
              };
            })()`,
          );
          if (after.missing) {
            row.hover = { ok: false, error: 'element gone after hover' };
          } else {
            row.hover = { ok: true, styles: after.styles };
            if (check.hoverScreenshot) {
              const png = await page.send('Page.captureScreenshot', {
                format: 'png',
                clip: clipFromBox(after.box, viewport),
                captureBeyondViewport: true,
              });
              const path = join(outDir, `${safeName(id)}-hover.png`);
              writeFileSync(path, Buffer.from(png.data, 'base64'));
              row.hover.screenshot = path;
            }
          }
        }
        results.push(row);
      } catch (e) {
        results.push({ id, ok: false, error: String(e.message || e).slice(0, 300) });
      }
    }

    process.stdout.write(
      JSON.stringify(
        {
          ok: true,
          browser: chrome,
          url: req.url,
          outDir,
          results,
        },
        null,
        2,
      ) + '\n',
    );
  } catch (e) {
    fail(String(e.message || e).slice(0, 400), { browser: chrome, results });
  } finally {
    try {
      page?.close();
    } catch {
    }
    try {
      if (child && child.exitCode == null) child.kill('SIGTERM');
    } catch {
    }
    await sleep(200);
    try {
      if (child && child.exitCode == null) child.kill('SIGKILL');
    } catch {
    }
  }
}

function safeName(id) {
  return String(id).replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 80) || 'check';
}

await main();
