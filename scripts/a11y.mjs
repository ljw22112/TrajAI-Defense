// Accessibility scan (axe-core) of the published pages: the technical report and the static console demo, in every
// console theme and at desktop and phone width. Fails on "serious" or "critical" violations (WCAG 2.1 A/AA rules).
//
// Usage: serve the repository under /TrajAI-Defense/ (as GitHub Pages does), then
//   node scripts/a11y.mjs http://127.0.0.1:8814/TrajAI-Defense/
// Requires web/node_modules (npm ci in web/): playwright and axe-core.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(new URL("../web/package.json", import.meta.url));
const { chromium } = require("playwright");
const AXE = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");

const base = (process.argv[2] || "http://127.0.0.1:8814/TrajAI-Defense/").replace(/\/?$/, "/");
const CONSOLE = ["", "scenarios", "scenarios/3", "scenarios/24", "raids/2", "compare?a=1&b=3", "analyses", "analyses/2",
  "analyses/11", "weather", "assets", "models"];
const THEMES = ["light", "dark", "military"];
const VIEWPORTS = [{ width: 1280, height: 800 }, { width: 390, height: 844 }];
const BLOCKING = new Set(["serious", "critical"]);

async function scan(page, url) {
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  await page.addScriptTag({ content: AXE });
  const r = await page.evaluate(() => window.axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } }));
  return r.violations;
}

const browser = await chromium.launch({ args: ["--no-sandbox"] });
const found = new Map();
let pages = 0;
for (const vp of VIEWPORTS) {
  for (const theme of [...THEMES, "report"]) {
    const ctx = await browser.newContext({ viewport: vp, colorScheme: theme === "dark" ? "dark" : "light" });
    const page = await ctx.newPage();
    const urls = theme === "report" ? [base] : CONSOLE.map(r => `${base}app/#/${r}`);
    if (theme !== "report") {
      await page.goto(`${base}app/`);
      await page.evaluate(t => { try { localStorage.setItem("trajai.theme", t); } catch { /* ignore */ } }, theme);
    }
    for (const url of urls) {
      pages++;
      for (const v of await scan(page, url)) {
        if (!BLOCKING.has(v.impact)) continue;
        const key = `${v.id} (${v.impact})`;
        const e = found.get(key) ?? { help: v.help, where: new Set() };
        for (const n of v.nodes.slice(0, 3)) e.where.add(`${theme} ${vp.width}px ${url.replace(base, "/")} :: ${n.target.join(" ")}`);
        found.set(key, e);
      }
    }
    await ctx.close();
  }
}
await browser.close();
console.log(`axe: ${pages} page scans`);
for (const [k, e] of found) {
  console.log(`\n${k}: ${e.help}`);
  [...e.where].slice(0, 8).forEach(w => console.log("  " + w));
}
console.log(found.size ? `\n${found.size} blocking rule(s) violated` : "no serious or critical violations");
process.exit(found.size ? 1 : 0);
