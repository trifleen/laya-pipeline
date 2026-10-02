const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (p) => (p * 100).toFixed(0) + "%";
let CFG = {};

/* ---------- tiny markdown (escaped first, so model output can't inject HTML) ---------- */
function md(src) {
  const blocks = [];
  src = src.replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => {
    blocks.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  // Display math ($$…$$ or \[…\]) has no renderer here: show it as a readable block.
  src = src.replace(/\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]/g, (_, a, b) => {
    blocks.push(`<pre class="math">${esc((a ?? b).trim())}</pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
  const out = []; let list = null, table = null, para = [];
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.t}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.t}>`); list = null; };
  const flushTable = () => { if (table) out.push(`<table>${table.map((r, i) => `<tr>${r.map((c) => i ? `<td>${inline(c)}</td>` : `<th>${inline(c)}</th>`).join("")}</tr>`).join("")}</table>`); table = null; };
  const flush = () => { flushPara(); flushList(); flushTable(); };
  for (const line of src.split("\n")) {
    let m;
    if ((m = line.match(/^\u0000(\d+)\u0000$/))) { flush(); out.push(blocks[+m[1]]); }
    else if ((m = line.match(/^(#{1,4})\s+(.*)/))) { flush(); out.push(`<h${m[1].length + 1}>${inline(m[2])}</h${m[1].length + 1}>`); }
    else if (/^\s*\|.*\|\s*$/.test(line)) {
      flushPara(); flushList();
      if (/^\s*\|[\s:|-]+\|\s*$/.test(line)) continue;
      (table ||= []).push(line.trim().slice(1, -1).split("|").map((c) => c.trim()));
    }
    else if ((m = line.match(/^\s*([-*•]|\d+\.)\s+(.*)/))) {
      flushPara(); flushTable();
      const t = /\d/.test(m[1]) ? "ol" : "ul";
      if (!list || list.t !== t) { flushList(); list = { t, items: [] }; }
      list.items.push(m[2]);
    }
    else if (/^\s*(---|\*\*\*)\s*$/.test(line)) { flush(); out.push("<hr>"); }
    else if (!line.trim()) flush();
    else { flushList(); flushTable(); para.push(line); }
  }
  flush();
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}


const fmtMs = (ms) => ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
const api = async (url, body, method) => {
  const res = await fetch(url, body === undefined && !method ? undefined
    : { method: method || "POST", headers: { "Content-Type": "application/json" }, body: body && JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
};

/* ---------- charts ----------
   Colours: the dataviz guide's validated categorical palette (slots 1-6, light + dark steps),
   always assigned in slot order. Marks: bars <= 14px with a 4px rounded data-end, 2px lines,
   8px markers with a 2px surface ring, hairline grid. Every mark has a hover tooltip. */
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

const tipEl = document.createElement("div");
tipEl.className = "tip";
document.body.appendChild(tipEl);
function showTip(e, html) {
  tipEl.innerHTML = html;
  tipEl.style.display = "block";
  const r = tipEl.getBoundingClientRect(), pad = 12;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tipEl.style.left = Math.max(8, x) + "px";
  tipEl.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { tipEl.style.display = "none"; };
function hover(el, html) {
  const show = (e) => showTip(e, typeof html === "function" ? html() : html);
  el.addEventListener("pointerenter", show);
  el.addEventListener("pointermove", show);
  el.addEventListener("pointerleave", hideTip);
}

const legend = (items) => `<div class="legend">${items.map((i) =>
  `<span><i style="background:${i.color}"></i>${esc(i.name)}</span>`).join("")}</div>`;

/** Horizontal bars, one series. rows: [{label, value, tip?}] */
function hbars(box, rows, { fmt = String, max } = {}) {
  if (!rows.length) { box.innerHTML = `<div class="empty">No data yet.</div>`; return; }
  const top = max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  box.innerHTML = `<div class="hbars">${rows.map((r, i) => `
    <span class="lbl">${esc(r.label)}</span>
    <div class="track" data-i="${i}"><i style="width:${(100 * r.value / top).toFixed(2)}%"></i></div>
    <span class="val">${esc(fmt(r.value))}</span>`).join("")}</div>`;
  box.querySelectorAll(".track").forEach((t) => {
    const r = rows[+t.dataset.i];
    hover(t, `<b>${esc(r.label)}</b><br>${esc(r.tip || fmt(r.value))}`);
  });
}

/** Vertical columns, one series. cols: [{label, value, tip?}]; every `labelEvery`th label shown. */
function columns(box, cols, { fmt = String, labelEvery = 1, height = 120 } = {}) {
  const top = Math.max(...cols.map((c) => c.value), 1e-9);
  box.innerHTML = `<div class="cols" style="height:${height}px">${cols.map((c, i) => `
    <div class="col-slot" data-i="${i}"><i style="height:${(100 * c.value / top).toFixed(2)}%"></i></div>`).join("")}</div>
    <div class="col-labels">${cols.map((c, i) => `<span>${i % labelEvery ? "" : esc(c.label)}</span>`).join("")}</div>`;
  box.querySelectorAll(".col-slot").forEach((s) => {
    const c = cols[+s.dataset.i];
    hover(s, `<b>${esc(c.label)}</b><br>${esc(c.tip || fmt(c.value))}`);
  });
}

/** Line chart. series: [{name, color, points: [{x, y, tip?}]}]. One y axis, always. */
function lineChart(box, series, { x: [x0, x1], y: [y0, y1], xFmt = String, yFmt = String,
  xTicks = 5, yTicks = 4, height = 190, diagonal = false, area = false, steps = false } = {}) {
  const W = Math.max(box.clientWidth, 260), H = height, m = { l: 42, r: series.length > 1 ? 70 : 14, t: 10, b: 24 };
  const sx = (v) => m.l + (W - m.l - m.r) * (v - x0) / ((x1 - x0) || 1);
  const sy = (v) => H - m.b - (H - m.t - m.b) * (v - y0) / ((y1 - y0) || 1);
  box.innerHTML = (series.length > 1 ? legend(series) : "");
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart" }, box);
  for (let i = 0; i <= yTicks; i++) {
    const v = y0 + (y1 - y0) * i / yTicks, y = sy(v);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg);
    svgEl("text", { x: m.l - 6, y: y + 4, "text-anchor": "end", class: "tick" }, svg).textContent = yFmt(v);
  }
  for (let i = 0; i <= xTicks; i++) {
    const v = x0 + (x1 - x0) * i / xTicks;
    svgEl("text", { x: sx(v), y: H - 6, "text-anchor": i === 0 ? "start" : i === xTicks ? "end" : "middle", class: "tick" }, svg).textContent = xFmt(v);
  }
  if (diagonal) svgEl("line", { x1: sx(x0), y1: sy(y0), x2: sx(x1), y2: sy(y1), class: "ref" }, svg);
  const ends = [];
  for (const s of series) {
    if (!s.points.length) continue;
    let d = "";
    s.points.forEach((p, i) => {
      const X = sx(p.x), Y = sy(p.y);
      if (i && steps) d += `H${X}`;
      d += `${i ? "L" : "M"}${X},${Y}`;
    });
    if (area) svgEl("path", { d: `${d}L${sx(s.points.at(-1).x)},${sy(y0)}L${sx(s.points[0].x)},${sy(y0)}Z`, fill: s.color, opacity: 0.1 }, svg);
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    if (s.markers !== false) for (const p of s.points) {
      svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: s.color, class: "dot" }, svg);
    }
    const last = s.points.at(-1);
    ends.push({ y: sy(last.y), x: sx(last.x), name: s.name });
  }
  // Direct end labels for 2-4 series; drop any that would collide (legend + tooltip still carry it).
  if (series.length > 1 && series.length <= 4) {
    const placed = [];
    for (const e of ends.sort((a, b) => a.y - b.y)) {
      if (placed.some((p) => Math.abs(p - e.y) < 13)) continue;
      placed.push(e.y);
      svgEl("text", { x: e.x + 8, y: e.y + 4, class: "endlbl" }, svg).textContent = e.name;
    }
  }
  // Hit targets larger than the marks.
  for (const s of series) for (const p of s.points) {
    const hit = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 11, fill: "transparent" }, svg);
    hover(hit, p.tip || `<b>${esc(s.name)}</b><br>${esc(xFmt(p.x))}: ${esc(yFmt(p.y))}`);
  }
  return svg;
}

/** Dots on a 0-1 probability axis with a threshold line. points: [{x, cat, tip}] */
function stripPlot(box, points, cats, { threshold, thresholdLabel = "", height = 92 } = {}) {
  const W = Math.max(box.clientWidth, 260), H = height, m = { l: 8, r: 8, t: 18, b: 22 };
  const sx = (v) => m.l + (W - m.l - m.r) * v;
  box.innerHTML = legend(cats.filter((c) => points.some((p) => p.cat === c.key)));
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart" }, box);
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "grid" }, svg);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("text", { x: sx(v), y: H - 6, "text-anchor": v === 0 ? "start" : v === 1 ? "end" : "middle", class: "tick" }, svg).textContent = pct(v);
  }
  const band = H - m.t - m.b - 10;
  points.forEach((p, i) => {
    const cat = cats.find((c) => c.key === p.cat);
    const y = m.t + 5 + band * ((i * 0.618034) % 1);  // golden-ratio jitter: stable, evenly spread
    const dot = svgEl("circle", { cx: sx(p.x), cy: y, r: 4, fill: cat.color, class: "dot" }, svg);
    const hit = svgEl("circle", { cx: sx(p.x), cy: y, r: 9, fill: "transparent" }, svg);
    hover(hit, p.tip);
    dot.dataset.i = i;
  });
  if (threshold != null) {
    const x = sx(threshold);
    svgEl("line", { x1: x, x2: x, y1: m.t - 6, y2: H - m.b, class: "thresh" }, svg);
    svgEl("text", { x, y: m.t - 8, "text-anchor": threshold > 0.8 ? "end" : threshold < 0.2 ? "start" : "middle", class: "tick strong" }, svg)
      .textContent = `${thresholdLabel} ${pct(threshold)}`;
  }
}

/** Re-render charts in `box` when it resizes (charts measure their width). */
function onResize(box, fn) {
  let w = box.clientWidth;
  new ResizeObserver(() => { if (box.clientWidth !== w && box.clientWidth) { w = box.clientWidth; fn(); } }).observe(box);
}
