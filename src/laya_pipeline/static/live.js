/* ---------- status bar ---------- */
async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    CFG = s.config;
    const lm = s.lmstudio;
    const loaded = lm.models.filter((m) => m.state === "loaded" && m.type !== "embeddings").map((m) => `${m.id.split("/").pop()} ${m.quant || ""}`);
    $("lm-dot").className = "dot " + (lm.up ? "ok" : "bad");
    $("lm-txt").textContent = lm.up ? (loaded.join(", ") || "no model loaded") : "server down — run: lms server start";
    $("laya-dot").className = "dot " + ({ready: "ok", loading: "warn", error: "bad"}[s.laya.state]);
    $("laya-txt").textContent = s.laya.state === "ready" ? `ready on ${s.laya.device}` : s.laya.state;
    if (s.gpu) {
      $("vram-bar").style.width = pct(s.gpu.used_mb / s.gpu.total_mb);
      $("vram-txt").textContent = `${(s.gpu.used_mb / 1024).toFixed(1)} / ${(s.gpu.total_mb / 1024).toFixed(1)} GB`;
      $("gpu-extra").textContent = `${s.gpu.util}% · ${s.gpu.temp_c}°C · ${Math.round(s.gpu.power_w)} W`;
    }
    $("lab-txt").textContent = `${s.labels.labelled} / ${s.labels.decisions}` + (s.labels.labelled ? ` · Laya wrong ${s.labels.laya_wrong}` : "");
    $("turns").textContent = s.turns ? `(${s.turns} turns)` : "";
  } catch { $("lm-dot").className = "dot bad"; $("lm-txt").textContent = "dashboard server unreachable"; }
}

/* ---------- pipeline flow ---------- */
const STAGES = [["compress", "Compress"], ["route", "Route"], ["generate", "Generate"], ["check", "Check"], ["retry", "Retry"]];
const stageSub = {};
function drawFlow(state = {}) {
  $("flow").innerHTML = STAGES.map(([k, n]) => {
    const s = state[k] || {};
    return `<div class="node ${s.status || ""}"><div class="name">${n}</div>
      <div class="sub">${esc(stageSub[k] || (s.status === "skipped" ? "skipped" : "—"))}</div>
      <div class="ms">${s.ms != null ? fmtMs(s.ms) : "&nbsp;"}</div></div>`;
  }).join("");
}

/* ---------- routing card ---------- */
function drawRoute(r) {
  const entries = Object.entries(r.task_probs).sort((a, b) => b[1] - a[1]);
  const th = r.thresholds;
  let h = `<div class="sub-h" style="margin-top:0">Task type</div><div class="bars">`;
  for (const [k, p] of entries) {
    const win = k === r.task;
    h += `<span class="lbl ${win ? "win" : ""}">${k}</span>
      <div class="bar"><i class="${win ? "win" : ""}" style="width:${pct(p)}"></i>${win ? `<span class="th" style="left:${pct(th.task)}" data-l="sure ≥ ${pct(th.task)}"></span>` : ""}</div>
      <span class="pct">${pct(p)}</span>`;
  }
  h += `</div><div class="sub-h">Needs a lot of thinking?</div><div class="bars">
    <span class="lbl ${r.hard ? "win" : ""}">hard</span>
    <div class="bar"><i class="${r.hard ? "hot" : ""}" style="width:${pct(r.p_hard)}"></i><span class="th" style="left:${pct(th.hard)}" data-l="think ≥ ${pct(th.hard)}"></span></div>
    <span class="pct">${pct(r.p_hard)}</span></div>
    <div class="verdict">
      <span class="tag model">${esc(r.model.split("/").pop())}</span>
      <span class="tag ${r.think ? "think" : ""}">${r.think ? "thinking on" : "thinking off"}</span>
      <span class="tag">temp ${r.temperature}</span>
      <span class="tag ${r.sure ? "" : "warn"}">${r.sure ? r.task : `unsure → neutral temp`}</span>
    </div>`;
  $("route").innerHTML = h;
  $("route-ms").textContent = `Laya ${r.laya_ms} ms`;
}

/* ---------- check card ---------- */
let checks = [];
function drawCheck() {
  $("check").innerHTML = checks.map((c) => `
    <div class="bars" style="margin-top:${c.attempt > 1 ? 14 : 4}px">
      <span class="lbl">${c.attempt > 1 ? "retry" : "answer"}</span>
      <div class="bar"><i class="${c.ok ? "okc" : "badc"}" style="width:${pct(c.p_bad)}"></i><span class="th" style="left:${pct(c.threshold)}" data-l="retry ≥ ${pct(c.threshold)}"></span></div>
      <span class="pct">${pct(c.p_bad)}</span>
    </div>
    <div class="verdict"><span class="tag ${c.ok ? "ok" : "bad"}">${c.ok ? "looks on-topic" : "flagged as off-topic"}</span>
    <span class="tag">p(bad answer) ${pct(c.p_bad)}</span></div>`).join("") || `<div class="empty">Checking…</div>`;
}

/* ---------- compression card ---------- */
let chunkData = [];
function drawComp(c) {
  chunkData = c.chunks;
  $("comp-note").textContent = c.note || "";
  // Similarities cluster tightly (e.g. 0.55-0.72), so stretch bar heights to the range present.
  const sims = c.chunks.map((x) => x.sim).filter((x) => x != null);
  const lo = Math.min(...sims), span = Math.max(Math.max(...sims) - lo, 1e-6);
  $("comp").innerHTML = `<div class="chunks">${c.chunks.map((x) => `
      <div class="chunk ${x.status}" data-i="${x.i}" title="#${x.i} ${x.status}${x.sim != null ? " · sim " + x.sim.toFixed(3) : ""}">
        <i style="height:${x.sim != null ? 10 + 90 * (x.sim - lo) / span : 100}%"></i><span>${x.i}</span></div>`).join("")}</div>
    <div class="legend"><span><i style="background:var(--ok)"></i>kept (embeddings)</span>
      <span><i style="background:var(--think)"></i>rescued by Laya</span>
      <span><i style="background:var(--warn)"></i>Laya said no</span>
      <span><i style="background:var(--faint)"></i>dropped</span><span>fill = similarity to the question</span></div>
    <div class="chunk-view" id="chunk-view"><span class="meta">Click a chunk to read it.</span></div>`;
  $("comp").querySelectorAll(".chunk").forEach((el) => el.onclick = () => {
    $("comp").querySelectorAll(".chunk").forEach((e) => e.classList.remove("sel"));
    el.classList.add("sel");
    const x = chunkData[+el.dataset.i];
    const meta = [`#${x.i}`, x.status, x.sim != null && `similarity ${x.sim.toFixed(3)} (rank ${x.rank + 1})`,
      x.p_relevant != null && `Laya p(relevant) ${pct(x.p_relevant)}`].filter(Boolean).join(" · ");
    $("chunk-view").innerHTML = `<div class="meta">${esc(meta)}</div>${esc(x.text)}`;
  });
}

/* ---------- running a question ---------- */
let running = false;
async function send() {
  const question = $("q").value.trim();
  if (!question || running) return;
  running = true; $("send").disabled = true; $("send").textContent = "Running…";
  const stages = {}; Object.keys(stageSub).forEach((k) => delete stageSub[k]);
  checks = []; drawFlow(stages);
  $("route").innerHTML = `<div class="empty">Laya is deciding…</div>`; $("route-ms").textContent = "";
  $("check").innerHTML = `<div class="empty">Waiting for the answer…</div>`;
  ["s-wait", "s-tps", "s-ans", "s-think"].forEach((id) => $(id).textContent = "–");
  if (!$("ctx").value.trim()) { $("comp").innerHTML = `<div class="empty">No context given — compression skipped.</div>`; $("comp-note").textContent = ""; }

  let answer = "", thinking = "", retried = false, nTok = 0, nThink = 0, genStart = 0;
  // Live timeline: stages grow while running; server-side offsets anchor them.
  let tl = { stages: [], first: {}, gpu: [], end_ms: 0 }, clientT0 = performance.now(), drawQueued = false;
  const drawTL = () => { if (drawQueued) return; drawQueued = true;
    requestAnimationFrame(() => { drawQueued = false; drawTimeline($("timeline"), tl); }); };
  const tlTimer = setInterval(() => {
    const now = performance.now() - clientT0;
    tl.end_ms = now;
    for (const st of tl.stages) if (st.running) st.ms = Math.max(0, now - st.start_ms);
    drawTL();
  }, 250);
  const box = $("answer");
  const render = (final) => {
    box.innerHTML = (thinking ? `<details class="thinking" ${final ? "" : "open"}><summary>Thinking · ${nThink} tokens</summary><div>${esc(thinking)}</div></details>` : "")
      + (retried ? `<div class="retry-note">First answer was flagged by Laya — retrying on the big model with thinking.</div>` : "")
      + `<div class="${final ? "" : "cursor"}">${md(answer)}</div>`;
    if (!final) { const t = box.querySelector(".thinking div"); if (t) t.scrollTop = t.scrollHeight; }
  };
  box.innerHTML = `<div class="cursor"></div>`;
  $("answer-meta").textContent = "";

  const handlers = {
    start: (d) => { $("run-id").textContent = "run " + d.id; clientT0 = performance.now(); },
    gpu: (d) => { tl.gpu.push(d); },
    stage: (d) => {
      stages[d.stage] = d;
      if (d.status === "running") tl.stages.push({ stage: d.stage, start_ms: d.start_ms, ms: 0, running: true });
      else if (d.ms != null) {
        const st = tl.stages.find((x) => x.stage === d.stage && x.running);
        if (st) Object.assign(st, { ms: d.ms, running: false });
      }
      drawTL();
      if (d.stage === "generate" && d.status === "running") stageSub.generate = "loading model…";
      if (d.stage === "retry" && d.status === "running") stageSub.retry = "big model + thinking";
      drawFlow(stages);
    },
    compress: (d) => { stageSub.compress = d.note.split(" (")[0]; drawComp(d); },
    route: (d) => { stageSub.route = `${d.sure ? d.task : "unsure"} · ${d.hard ? "hard" : "easy"}`; stageSub.generate = d.model.split("/").pop(); drawRoute(d); drawFlow(stages); },
    model_loaded: (d) => { tl.first[retried ? "retry" : "generate"] = d.wait_ms; $("s-wait").textContent = fmtMs(d.wait_ms); genStart = performance.now(); stageSub[retried ? "retry" : "generate"] = d.model.split("/").pop(); drawFlow(stages); },
    reasoning: (d) => { thinking += d.text; nThink++; $("s-think").textContent = nThink; tick(); render(false); },
    token: (d) => { answer += d.text; nTok++; $("s-ans").textContent = nTok; tick(); render(false); },
    check: (d) => { checks.push(d); stageSub[d.attempt > 1 ? "retry" : "check"] = d.ok ? "on-topic" : "flagged"; drawCheck(); drawFlow(stages); },
    retry: () => { retried = true; answer = ""; thinking = ""; nTok = 0; nThink = 0; render(false); },
    done: (d) => {
      answer = d.answer; render(true);
      clearInterval(tlTimer); tl = traceTimeline(d.trace); drawTL();
      const g = d.trace.retry || d.trace.generate;
      if (g) { $("s-tps").textContent = g.tokens_per_s; $("s-ans").textContent = g.answer_tokens; $("s-think").textContent = g.reasoning_tokens; }
      $("answer-meta").textContent = `${d.trace.seconds} s total`;
    },
    error: (d) => { box.innerHTML += `<div class="err">${esc(d.message)}</div>`; },
  };
  const tick = () => { if (genStart) { const s = (performance.now() - genStart) / 1000; if (s > .3) $("s-tps").textContent = ((nTok + nThink) / s).toFixed(1); } };

  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context: $("ctx").value, remember: $("remember").checked }) });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = raw.match(/^event: (.*)$/m)?.[1], data = raw.match(/^data: (.*)$/m)?.[1];
        if (ev && handlers[ev]) handlers[ev](JSON.parse(data));
      }
    }
    $("q").value = "";
  } catch (e) {
    box.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    clearInterval(tlTimer);
    running = false; $("send").disabled = false; $("send").textContent = "Send";
    refreshStatus(); loadHistory();
  }
}

/* ---------- history + labelling ---------- */
let HIST = { runs: [], options: {} }; const open = new Set();
const QNAMES = { task: "task", hard: "thinking", answers: "check" };
const OPTNAMES = { hard: { A: "hard", B: "easy" }, answers: { A: "good", B: "bad" } };
async function loadHistory() {
  HIST = await (await fetch("/api/history")).json();
  drawHistory();
}
function drawHistory() {
  if (!HIST.runs.length) { $("hist").innerHTML = `<tr><td class="empty">No runs yet.</td></tr>`; return; }
  let h = `<tr><th>Time</th><th>Question</th><th>Route</th><th>Model</th><th>Check</th><th>Time</th></tr>`;
  for (const r of HIST.runs) {
    const t = r.trace, ro = t.route || {}, ck = t.checks || [];
    const okLast = ck.length ? ck[ck.length - 1].ok : null;
    h += `<tr class="run" data-id="${t.id}">
      <td class="num">${esc(r.time.slice(11, 16))}</td>
      <td class="q" title="${esc(r.question)}">${esc(r.question)}</td>
      <td>${esc(ro.task || "?")}${ro.hard ? ' <span class="tag think">hard</span>' : ""}</td>
      <td class="num">${esc((ro.model || "").split("/").pop())}${t.retried_with ? " ↻" : ""}</td>
      <td>${okLast == null ? "" : `<span class="tag ${okLast ? "ok" : "bad"}">${okLast ? "ok" : "flag"}</span>`}</td>
      <td class="num">${t.seconds ?? ""} s</td></tr>`;
    if (open.has(t.id)) {
      h += `<tr><td colspan="6"><div class="labeler">
        <div class="answer-preview">${esc(r.answer.slice(0, 1200))}</div>
        ${t.stages ? `<div class="hist-tl" data-run="${t.id}"></div>` : ""}
        ${r.decisions.map((d) => `<div class="item"><span class="qn">${QNAMES[d.question] || d.question}</span>
          ${HIST.options[d.question].map((o) => {
            const cls = ["opt", o === d.predicted ? "pred" : "", d.label === o ? "chosen" : "", d.label === o && o !== d.predicted ? "fix" : ""].join(" ");
            const name = (OPTNAMES[d.question] || {})[o] || o;
            return `<button class="${cls}" data-d="${d.id}" data-v="${o}" title="Laya: ${pct(d.probabilities[o])}">${name} <span class="kbd">${pct(d.probabilities[o])}</span></button>`;
          }).join("")}
          <span class="kbd">${d.label == null ? "unlabelled — click the correct one" : d.label === d.predicted ? "✓ Laya was right" : "✗ corrected"}</span></div>`).join("")}
      </div></td></tr>`;
    }
  }
  $("hist").innerHTML = h;
  $("hist").querySelectorAll(".hist-tl").forEach((el) => {
    const run = HIST.runs.find((r) => r.trace.id === el.dataset.run);
    drawTimeline(el, traceTimeline(run.trace));
  });
  $("hist").querySelectorAll("tr.run").forEach((tr) => tr.onclick = () => {
    open.has(tr.dataset.id) ? open.delete(tr.dataset.id) : open.add(tr.dataset.id); drawHistory();
  });
  $("hist").querySelectorAll(".opt").forEach((b) => b.onclick = async (e) => {
    e.stopPropagation();
    const res = await fetch("/api/label", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: b.dataset.d, value: b.dataset.v }) });
    if (res.ok) { for (const r of HIST.runs) for (const d of r.decisions) if (d.id === b.dataset.d) d.label = b.dataset.v;
      drawHistory(); refreshStatus(); }
  });
}

/* ---------- wiring ---------- */
$("send").onclick = send;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send(); });
$("reset").onclick = async () => { await fetch("/api/reset", { method: "POST" }); refreshStatus(); };
const ctxInfo = () => { const n = $("ctx").value.length; $("ctx-info").textContent = n ? `· ${n.toLocaleString()} chars` : ""; };
$("ctx").addEventListener("input", ctxInfo);
$("ctx-clear").onclick = () => { $("ctx").value = ""; $("files").value = ""; ctxInfo(); };
$("files").onchange = async () => {
  const texts = await Promise.all([...$("files").files].map(async (f) => `# ${f.name}\n\n${await f.text()}`));
  $("ctx").value = [$("ctx").value.trim(), ...texts].filter(Boolean).join("\n\n");
  ctxInfo();
};
drawFlow(); refreshStatus(); loadHistory();
setInterval(refreshStatus, 2000);

/* ---------- timeline ---------- */
const SEG = {  // fixed slot order = fixed colours
  compress: { name: "Compress", color: "var(--s1)" },
  route: { name: "Laya route", color: "var(--s2)" },
  load: { name: "Model load + prompt", color: "var(--s3)" },
  generate: { name: "Generating", color: "var(--s4)" },
  check: { name: "Laya check", color: "var(--s5)" },
  retry: { name: "Retry generating", color: "var(--s6)" },
};

/** t: {stages: [{stage, start_ms, ms}], first: {generate, retry}, gpu: [...], end_ms} */
function drawTimeline(box, t) {
  const end = Math.max(t.end_ms || 0, ...t.stages.map((s) => s.start_ms + s.ms), 1);
  const segs = [];
  for (const s of t.stages) {
    const first = t.first?.[s.stage];
    if ((s.stage === "generate" || s.stage === "retry") && first != null) {
      segs.push({ key: "load", start: s.start_ms, ms: Math.min(first, s.ms) });
      if (s.ms > first) segs.push({ key: s.stage, start: s.start_ms + first, ms: s.ms - first });
    } else if (s.stage === "retry") {
      segs.push({ key: "load", start: s.start_ms, ms: s.ms });
    } else segs.push({ key: s.stage, start: s.start_ms, ms: s.ms });
  }
  const used = [...new Set(segs.map((s) => s.key))];
  const ticks = niceTicks(end / 1000);
  box.innerHTML = `${legend(used.map((k) => SEG[k]))}
    <div class="tl-track">${segs.map((s, i) => {
      const left = 100 * s.start / end, width = 100 * s.ms / end;
      const fits = (width / 100) * (box.clientWidth || 600) > SEG[s.key].name.length * 6.5 + 14;
      return `<div class="tl-seg k-${s.key}" data-i="${i}" style="left:${left}%;width:calc(${width}% - 2px);background:${SEG[s.key].color}">
        ${fits ? `<span>${esc(SEG[s.key].name)} · ${fmtMs(s.ms)}</span>` : ""}</div>`;
    }).join("")}</div>
    <div class="tl-axis">${ticks.map((v) => `<span style="left:${100 * v * 1000 / end}%">${v}s</span>`).join("")}</div>
    <div class="tl-gpu"><div><div class="sub-h">VRAM (GB)</div><div class="tl-vram"></div></div>
      <div><div class="sub-h">GPU power (W)</div><div class="tl-power"></div></div></div>`;
  box.querySelectorAll(".tl-seg").forEach((el) => {
    const s = segs[+el.dataset.i];
    hover(el, `<b>${esc(SEG[s.key].name)}</b><br>${fmtMs(s.ms)} · starts at ${fmtMs(s.start)}`);
  });
  const gpu = t.gpu || [];
  const gb = (mb) => mb / 1024;
  const common = { x: [0, end / 1000], xFmt: (v) => v.toFixed(v < 10 ? 1 : 0) + "s", height: 110, area: true, xTicks: 4, yTicks: 2 };
  if (gpu.length > 1) {
    const maxV = Math.max(8, ...gpu.map((g) => gb(g.vram_mb)));
    lineChart(box.querySelector(".tl-vram"), [{ name: "VRAM", color: "var(--s1)", markers: false,
      points: gpu.map((g) => ({ x: g.t_ms / 1000, y: gb(g.vram_mb), tip: `${(g.t_ms / 1000).toFixed(1)}s · ${gb(g.vram_mb).toFixed(2)} GB · load ${g.util}%` })) }],
      { ...common, y: [0, Math.ceil(maxV)], yFmt: (v) => v.toFixed(0) });
    const maxP = Math.max(20, ...gpu.map((g) => g.power_w));
    lineChart(box.querySelector(".tl-power"), [{ name: "Power", color: "var(--s1)", markers: false,
      points: gpu.map((g) => ({ x: g.t_ms / 1000, y: g.power_w, tip: `${(g.t_ms / 1000).toFixed(1)}s · ${Math.round(g.power_w)} W` })) }],
      { ...common, y: [0, Math.ceil(maxP / 20) * 20], yFmt: (v) => v.toFixed(0) });
  } else {
    box.querySelectorAll(".tl-vram, .tl-power").forEach((el) => el.innerHTML = `<div class="empty">No GPU samples.</div>`);
  }
}

function niceTicks(maxS) {
  const step = [0.5, 1, 2, 5, 10, 20, 30, 60].find((s) => maxS / s <= 6) || 120;
  const out = [];
  for (let v = 0; v <= maxS + 1e-9; v += step) out.push(+v.toFixed(1));
  return out;
}

const traceTimeline = (trace) => ({
  stages: trace.stages || [], gpu: trace.gpu || [], end_ms: (trace.seconds || 0) * 1000,
  first: { generate: trace.generate?.first_token_ms, retry: trace.retry?.first_token_ms },
});
