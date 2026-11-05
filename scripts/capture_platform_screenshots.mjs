#!/usr/bin/env node
/**
 * Capture Persian (fa) screenshots of all eight platform pages.
 *
 * Used as pull-request evidence for the `make-platform-bilingual-and-audit-page-bugs`
 * change (task 9.2/9.3): reviewers can see the RTL layout, Persian typography,
 * translated labels, and the data-driven controls before merging.
 *
 * The script is deliberately dependency-free apart from `puppeteer-core`, and it
 * drives *headless Chrome over the DevTools protocol*: no npm install is needed
 * in `platform/` (the platform itself gains no new runtime dependency), and the
 * caller supplies the browser binary —
 *
 *   node scripts/capture_platform_screenshots.mjs \
 *     --executable /path/to/chrome \
 *     [--out reports/screenshots/fa] [--locale fa] [--width 1680] [--height 1500]
 *
 * The build is a single self-contained HTML file (`platform/dist/index.html`),
 * but its inlined bundle is an ES module, and `file://` module loading is
 * blocked by browsers ("cross-origin requests are only supported for http(s)").
 * The script therefore starts a throwaway static server on 127.0.0.1 for the
 * duration of the capture — still fully offline (fonts are self-hosted, no CDN).
 *
 * If no Chrome/Chromium binary is available (e.g. a slim CI container), the
 * script exits 2 with an explanatory message; the manual fallback is:
 *   1. `cd platform && npm run build`
 *   2. open `platform/dist/index.html`, click FA, and screenshot each nav entry.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DIST = path.join(ROOT, "platform", "dist", "index.html");

// ── CLI ──────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const out = { out: path.join(ROOT, "reports", "screenshots", "fa"), locale: "fa", width: 1680, height: 1500, port: 0 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--executable") out.executable = next();
    else if (a === "--out") out.out = path.resolve(next());
    else if (a === "--locale") out.locale = next();
    else if (a === "--width") out.width = Number(next());
    else if (a === "--height") out.height = Number(next());
    else if (a === "--port") out.port = Number(next());
    else if (a === "--help" || a === "-h") {
      console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0]);
      process.exit(0);
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

const PAGES = [
  { id: "overview", file: "01_overview.png" },
  { id: "map", file: "02_geospatial-explorer.png" },
  { id: "temporal", file: "03_temporal-analysis.png" },
  { id: "spectral", file: "04_spectral-temporal.png" },
  { id: "validation", file: "05_validation-lab.png" },
  { id: "comparison", file: "06_model-comparison.png" },
  { id: "spatial", file: "07_spatial-analysis.png" },
  { id: "reports", file: "08_reports-experiments.png" },
];

// ── Browser discovery ────────────────────────────────────────────────────────
function findExecutable() {
  if (args.executable) return args.executable;
  const candidates = [
    process.env.CHROME_PATH,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
  ].filter(Boolean);
  for (const c of candidates) {
    try {
      if (c && fs.existsSync(c)) return c;
    } catch {
      /* ignore */
    }
  }
  return null;
}

// ── Minimal DevTools-protocol client (no puppeteer, no websocket package) ─────
async function connect(webSocketDebuggerUrl) {
  const { WebSocket } = globalThis; // Node ≥ 22 ships a built-in WebSocket client
  const ws = new WebSocket(webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error("failed to open DevTools websocket"));
  });
  let id = 0;
  const pending = new Map();
  const events = new Map();
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(`${msg.error.message} (${msg.error.code})`)) : resolve(msg.result);
    } else if (msg.method) {
      const list = events.get(msg.method) ?? [];
      events.set(msg.method, []);
      list.forEach(fn => fn(msg.params));
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const msgId = ++id;
      pending.set(msgId, { resolve, reject });
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  const once = method =>
    new Promise(resolve => {
      const list = events.get(method) ?? [];
      list.push(resolve);
      events.set(method, list);
    });
  return { send, once, close: () => ws.close() };
}

function rpcHttp(port, method, pathname) {
  return new Promise((resolve, reject) => {
    const req = http.request({ port, method, path: pathname }, res => {
      let body = "";
      res.on("data", d => (body += d));
      res.on("end", () => {
        try {
          resolve(JSON.parse(body));
        } catch (err) {
          reject(new Error(`bad response for ${pathname}: ${body.slice(0, 200)}`));
        }
      });
    });
    req.on("error", reject);
    req.end();
  });
}

async function waitForDevTools(port, timeoutMs = 30000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const list = await rpcHttp(port, "GET", "/json/list");
      if (Array.isArray(list) && list.length) return list;
    } catch {
      /* not up yet */
    }
    await new Promise(r => setTimeout(r, 200));
  }
  throw new Error(`Chrome DevTools endpoint on port ${port} did not respond within ${timeoutMs} ms`);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

/** Throwaway static file server for platform/dist (127.0.0.1 only). */
function serveDist() {
  const dir = path.dirname(DIST);
  const mime = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".json": "application/json", ".png": "image/png" };
  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent((req.url || "/").split("?")[0]);
    const file = path.join(dir, rel === "/" ? "index.html" : rel);
    if (!file.startsWith(dir)) {
      res.writeHead(403).end();
      return;
    }
    fs.readFile(file, (err, buf) => {
      if (err) {
        res.writeHead(404).end("not found");
        return;
      }
      res.writeHead(200, { "content-type": mime[path.extname(file)] ?? "application/octet-stream" });
      res.end(buf);
    });
  });
  return new Promise(resolve => server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port })));
}

async function main() {
  if (!fs.existsSync(DIST)) {
    console.error(`[capture] missing build: ${DIST}\n          run:  cd platform && npm run build`);
    process.exit(1);
  }
  const executable = findExecutable();
  if (!executable) {
    console.error(
      "[capture] no Chrome/Chromium binary found.\n" +
        "          pass --executable /path/to/chrome (or set CHROME_PATH).\n" +
        "          Manual fallback: open platform/dist/index.html, switch to FA and screenshot each page."
    );
    process.exit(2);
  }

  fs.mkdirSync(args.out, { recursive: true });
  const { server, port: httpPort } = await serveDist();
  console.log(`[capture] serving ${path.relative(ROOT, path.dirname(DIST))} on 127.0.0.1:${httpPort}`);
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "agri-shots-"));
  const port = args.port || 9200 + Math.floor(Math.random() * 400);

  const child = spawn(
    executable,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-dev-shm-usage",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${userDataDir}`,
      `--window-size=${args.width},${args.height}`,
      "about:blank",
    ],
    { stdio: ["ignore", "pipe", "pipe"] }
  );
  const chromeLog = [];
  child.stderr.on("data", d => chromeLog.push(String(d)));
  child.stdout.on("data", d => chromeLog.push(String(d)));

  let session;
  try {
    await waitForDevTools(port);
    const targets = await rpcHttp(port, "GET", "/json/new?about:blank", undefined);
    void targets;
  } catch {
    /* /json/new is blocked on some builds; reuse the initial page target below */
  }

  try {
    const list = await waitForDevTools(port);
    const pageTarget = list.find(t => t.type === "page") ?? list[0];
    session = await connect(pageTarget.webSocketDebuggerUrl);
    const { send, once } = session;

    await send("Page.enable");
    await send("Runtime.enable");
    await send("Emulation.setDeviceMetricsOverride", {
      width: args.width,
      height: args.height,
      deviceScaleFactor: 2,
      mobile: false,
    });

    // Persist the locale *before* the app's first script runs, exactly like a
    // returning visitor — this exercises the localStorage-restore path of the
    // `Persistent language preference` requirement rather than a UI click.
    await send("Page.addScriptToEvaluateOnNewDocument", {
      source: `try { localStorage.setItem("agrianomaly.locale", ${JSON.stringify(args.locale)}); } catch (e) {}`,
    });

    const url = `http://127.0.0.1:${httpPort}/index.html`;
    await send("Page.navigate", { url });

    // Poll instead of trusting load events: the inlined bundle is ~9 MB and
    // React mounts asynchronously after `load`.
    const probe = () =>
      send("Runtime.evaluate", {
        expression: `JSON.stringify({
          ready: document.readyState,
          dir: document.documentElement.dir,
          lang: document.documentElement.lang,
          nav: document.querySelectorAll('aside nav button').length,
          localized: document.body.innerText.includes(${JSON.stringify(navLabels(args.locale).overview)}),
          font: getComputedStyle(document.body).fontFamily,
        })`,
        returnByValue: true,
      }).then(r => {
        // The first probe can race the navigation and come back undefined.
        try {
          return JSON.parse(r?.result?.value ?? "{}");
        } catch {
          return {};
        }
      });

    let state = await probe();
    for (let i = 0; i < 120 && !(state.nav >= PAGES.length && state.dir === (args.locale === "fa" ? "rtl" : "ltr")); i++) {
      await sleep(250);
      state = await probe();
    }
    if (state.nav < PAGES.length) throw new Error(`app did not mount: ${JSON.stringify(state)}`);
    if (state.dir !== (args.locale === "fa" ? "rtl" : "ltr") || state.lang !== args.locale) {
      throw new Error(`locale not applied: ${JSON.stringify(state)}`);
    }
    if (!state.localized) throw new Error(`copy not translated to ${args.locale}: ${JSON.stringify(state)}`);
    console.log(`[capture] mounted · dir=${state.dir} · lang=${state.lang} · nav=${state.nav} buttons · font=${state.font.split(",")[0]}`);

    // Wait for the webfont to be ready so Persian text is not captured in a
    // fallback face (font-display: swap would otherwise show Inter/system).
    await send("Runtime.evaluate", { expression: "document.fonts.ready.then(() => true)", awaitPromise: true });
    await sleep(600);

    for (const page of PAGES) {
      const clicked = await send("Runtime.evaluate", {
        expression: `(() => {
          const buttons = [...document.querySelectorAll('aside nav button')];
          const target = buttons.find(b => (b.textContent || '').includes(${JSON.stringify(navLabels(args.locale)[page.id])}));
          if (!target) return false;
          target.click();
          return true;
        })()`,
        returnByValue: true,
      });
      if (!clicked.result.value) throw new Error(`could not activate nav item "${page.id}"`);
      // Wait until the clicked nav entry is marked active, then let charts
      // animate in and Leaflet lay out its tiles.
      for (let i = 0; i < 40; i++) {
        const active = await send("Runtime.evaluate", {
          expression: `(() => { const b = [...document.querySelectorAll('aside nav button')].find(x => x.getAttribute('aria-current') === 'page'); return b ? b.textContent : ''; })()`,
          returnByValue: true,
        });
        if ((active.result.value || "").includes(navLabels(args.locale)[page.id])) break;
        await sleep(150);
      }
      await sleep(1500); // chart animations + Leaflet tile layout

      // Size the viewport to the *full* page, wait for Recharts'
      // ResponsiveContainer to re-measure, and only then capture. Using
      // `captureBeyondViewport` instead produces mid-reflow frames where
      // charts are briefly 14×14 px (Recharts' minimum surface).
      const height = await send("Runtime.evaluate", { expression: "document.documentElement.scrollHeight", returnByValue: true });
      const fullHeight = Math.max(args.height, Math.ceil(height.result.value ?? args.height));
      await send("Emulation.setDeviceMetricsOverride", { width: args.width, height: fullHeight, deviceScaleFactor: 2, mobile: false });
      await sleep(1200);
      const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
      const target = path.join(args.out, page.file);
      fs.writeFileSync(target, Buffer.from(data, "base64"));
      console.log(`[capture] ${path.relative(ROOT, target)}  (${args.width}×${fullHeight} css)`);
      // Reset for the next page so its layout is measured at the base viewport.
      await send("Emulation.setDeviceMetricsOverride", { width: args.width, height: args.height, deviceScaleFactor: 2, mobile: false });
      await sleep(300);
      const { result } = await send("Runtime.evaluate", { expression: "document.documentElement.scrollHeight", returnByValue: true });
    }

    fs.writeFileSync(path.join(args.out, "README.md"), README_TEXT(args));
    console.log(`[capture] done → ${path.relative(ROOT, args.out)}`);
  } finally {
    try {
      server.close();
    } catch {
      /* ignore */
    }
    try {
      session?.close();
    } catch {
      /* ignore */
    }
    child.kill("SIGTERM");
    await new Promise(r => setTimeout(r, 400));
    try {
      fs.rmSync(userDataDir, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
    if (chromeLog.length && process.env.DEBUG_CHROME) console.error(chromeLog.join(""));
  }
}

/** Nav labels per locale, used to locate each page's sidebar button. */
const NAV_LABELS_EN = {
  overview: "Overview",
  map: "Geospatial Explorer",
  temporal: "Temporal Analysis",
  spectral: "Spectral",
  validation: "Validation Lab",
  comparison: "Model Comparison",
  spatial: "Spatial Analysis",
  reports: "Reports & Experiments",
};

const NAV_LABELS_FA = {
  overview: "نمای کلی",
  map: "کاوشگر جغرافیایی",
  temporal: "تحلیل زمانی",
  spectral: "طیفی–زمانی",
  validation: "آزمایشگاه اعتبارسنجی",
  comparison: "مقایسه مدل‌ها",
  spatial: "تحلیل فضایی",
  reports: "گزارش‌ها و آزمایش‌ها",
};

const navLabels = locale => (locale === "fa" ? NAV_LABELS_FA : NAV_LABELS_EN);

const README_TEXT = a => `# Persian platform screenshots (fa)

Generated by \`node scripts/capture_platform_screenshots.mjs --locale fa\`.

| File | Page | Route id |
| --- | --- | --- |
${PAGES.map((p, i) => `| \`${p.file}\` | ${i + 1}. ${p.id} | \`${p.id}\` |`).join("\n")}

- Viewport: ${a.width}×${a.height} CSS px at deviceScaleFactor 2.
- Locale restored from \`localStorage\` (\`agrianomaly.locale = ${a.locale}\`) before first paint,
  so \`<html dir="rtl" lang="fa">\` and the Vazirmatn font stack are active in every frame.
- Source: the single-file production build (\`platform/dist/index.html\`), loaded over \`file://\`.
- Regenerate: \`cd platform && npm run build && cd .. && node scripts/capture_platform_screenshots.mjs --executable /path/to/chrome\`.
`;

main().catch(err => {
  console.error(`[capture] ${err.stack || err.message}`);
  process.exit(1);
});
