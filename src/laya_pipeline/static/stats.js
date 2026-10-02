/* Stats tab: routing, speed, Laya's accuracy as you label, and when you use it. */
const Stats = (() => {
  const QN = { task: "task", hard: "thinking", answers: "check", relevant: "relevance" };
  const QCOLOR = { task: "var(--s1)", hard: "var(--s2)", answers: "var(--s3)", relevant: "var(--s4)" };
  let S = null;
  const root = () => $("tab-stats");
  const short = (m) => (m || "?").split("/").pop();
  const avg = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
  const median = (xs) => { if (!xs.length) return 0; const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
  const countBy = (xs, f) => xs.reduce((m, x) => { const k = f(x); if (k != null) m[k] = (m[k] || 0) + 1; return m; }, {});

  async function show() {
    S = await api("/api/stats");
    render();
  }

  function render() {
    const runs = S.runs, lab = S.labelled;
    if (!runs.length) {
      root().innerHTML = `<div class="col wide"><section class="card"><div class="empty">No runs yet — ask something in the Live tab.</div></section></div>`;
      return;
    }
    const right = lab.filter((l) => l.correct).length;
    const comp = runs.filter((r) => r.compressed?.chunks_in);
    root().innerHTML = `
      <div class="col wide">
        <section class="card"><div class="stats tiles">
          <div class="stat"><div class="v">${runs.length}</div><div class="k">runs</div></div>
          <div class="stat"><div class="v">${median(runs.map((r) => r.seconds || 0)).toFixed(1)} s</div><div class="k">median time per answer</div></div>
          <div class="stat"><div class="v">${pct(runs.filter((r) => r.hard).length / runs.length)}</div><div class="k">used thinking</div></div>
          <div class="stat"><div class="v">${pct(runs.filter((r) => r.retried).length / runs.length)}</div><div class="k">retried after check</div></div>
          <div class="stat"><div class="v">${lab.length ? pct(right / lab.length) : "–"}</div><div class="k">Laya right · ${lab.length} labelled</div></div>
          <div class="stat"><div class="v">${comp.length ? pct(avg(comp.map((r) => r.compressed.chunks_out / r.compressed.chunks_in))) : "–"}</div><div class="k">context kept when compressing</div></div>
        </div></section>
        <div class="lab-grid">
          <section class="card"><h2>Task types <small>what Laya decided</small></h2><div id="st-task"></div></section>
          <section class="card"><h2>Model that answered <small>after retries</small></h2><div id="st-model"></div></section>
          <section class="card"><h2>Speed by model <small>average tokens per second</small></h2><div id="st-tps"></div></section>
          <section class="card"><h2>Wait for first token <small>average, includes model loading</small></h2><div id="st-first"></div></section>
        </div>
        <section class="card"><h2>Laya's accuracy as you label <small>running share right, per question</small></h2><div id="st-acc"></div></section>
        <section class="card"><h2>Busiest hours <small>runs per hour of day</small></h2><div id="st-hours"></div></section>
      </div>`;
    draw();
    onResizeOnce();
  }

  function draw() {
    const runs = S.runs;
    const tasks = countBy(runs, (r) => r.task);
    hbars($("st-task"), Object.entries(tasks).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
      ({ label: k, value: v, tip: `${v} runs · ${pct(v / runs.length)}` })), { fmt: String });
    const models = countBy(runs, (r) => short(r.final_model || r.model));
    hbars($("st-model"), Object.entries(models).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
      ({ label: k, value: v, tip: `${v} runs · ${pct(v / runs.length)}` })), { fmt: String });

    const byModel = {};
    for (const r of runs) (byModel[short(r.final_model || r.model)] ||= []).push(r);
    const speed = Object.entries(byModel).map(([m, rs]) => {
      const t = rs.filter((r) => r.tps).map((r) => r.tps);
      return { label: m, value: avg(t), tip: `${avg(t).toFixed(1)} tokens/s over ${t.length} runs` };
    }).filter((x) => x.value);
    hbars($("st-tps"), speed, { fmt: (v) => v.toFixed(1) });
    const first = Object.entries(byModel).map(([m, rs]) => {
      const t = rs.filter((r) => r.first_token_ms != null).map((r) => r.first_token_ms);
      return { label: m, value: avg(t) / 1000, tip: `${fmtMs(avg(t))} average over ${t.length} runs (median ${fmtMs(median(t))})` };
    }).filter((x) => x.value);
    hbars($("st-first"), first, { fmt: (v) => v.toFixed(1) + " s" });

    // Running accuracy per question over labelled decisions, in label order.
    const lab = S.labelled;
    const accBox = $("st-acc");
    if (lab.length < 2) accBox.innerHTML = `<div class="empty">Label decisions in the Live tab's History to track how often Laya is right.</div>`;
    else {
      const series = Object.keys(QN).map((q) => {
        const pts = []; let ok = 0, n = 0;
        lab.forEach((l, i) => {
          if (l.question !== q) return;
          n++; ok += l.correct;
          pts.push({ x: i + 1, y: ok / n, tip: `<b>${QN[q]}</b> · label #${i + 1}<br>${ok}/${n} right (${pct(ok / n)})` });
        });
        return { name: QN[q], color: QCOLOR[q], points: pts };
      }).filter((s) => s.points.length);
      lineChart(accBox, series, { x: [1, lab.length], y: [0, 1], xFmt: (v) => Math.round(v), yFmt: pct, height: 220 });
    }

    const hours = Array(24).fill(0);
    for (const r of runs) hours[+r.time.slice(11, 13)]++;
    columns($("st-hours"), hours.map((v, h) => ({ label: String(h).padStart(2, "0"), value: v,
      tip: `${String(h).padStart(2, "0")}:00–${String(h).padStart(2, "0")}:59 · ${v} runs` })), { labelEvery: 3, height: 110 });
  }

  let watching = false;
  function onResizeOnce() {
    if (watching) return;
    watching = true;
    onResize(root(), () => { if (S && !root().hidden) draw(); });
  }

  return { show };
})();
