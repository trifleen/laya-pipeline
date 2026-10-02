/* Lab tab: what-if thresholds, calibration, and the question editor. */
const Lab = (() => {
  const QN = { task: "Task type", hard: "Needs thinking", answers: "Answer check", relevant: "Chunk relevance", scope: "Context scope" };
  const OPT = { hard: { A: "hard", B: "easy" }, answers: { A: "good", B: "bad" }, relevant: { A: "relevant", B: "not relevant" }, scope: { A: "one passage", B: "a few sections", C: "most of it" } };
  const optName = (q, k) => OPT[q]?.[k] || k;

  // How each threshold reads a decision: x = the probability compared to the threshold,
  // yes = "a labelled example that SHOULD land at or above the threshold".
  const SLIDERS = {
    TASK_MIN_PROB: { q: "task", title: "Trust the task guess", what: "Laya's top task probability. Below → unsure → neutral temperature.",
      x: (p) => Math.max(...Object.values(p)), yes: (d, p) => d.label === top(p), yesName: "Laya's guess was right", noName: "Laya's guess was wrong" },
    HARD_MIN_PROB: { q: "hard", title: "Turn thinking on", what: "p(hard). At or above → Qwen with thinking.",
      x: (p) => p.A, yes: (d) => d.label === "A", yesName: "labelled hard", noName: "labelled easy" },
    BAD_ANSWER_MIN_PROB: { q: "answers", title: "Retry an answer", what: "p(bad answer). At or above → retry on Qwen with thinking.",
      x: (p) => p.B, yes: (d) => d.label === "B", yesName: "labelled bad", noName: "labelled good" },
    RELEVANCE_MIN_PROB: { q: "relevant", title: "Rescue a chunk", what: "p(relevant) for borderline chunks. At or above → sent to the model.",
      x: (p) => p.A, yes: (d) => d.label === "A", yesName: "labelled relevant", noName: "labelled not relevant" },
    SCOPE_MIN_PROB: { q: "scope", title: "Trust the context budget", what: "Laya's top scope probability. At or above → 3 / 6 / 12 chunks; below → the default 6.",
      x: (p) => Math.max(...Object.values(p)), yes: (d, p) => d.label === top(p), yesName: "Laya's guess was right", noName: "Laya's guess was wrong" },
  };
  const top = (p) => Object.keys(p).reduce((a, b) => (p[a] >= p[b] ? a : b));

  let D = null, DEC = [], TH = {}, calQ = "hard", calRes = null, edQ = "hard", evalRes = null, busy = false;
  const root = () => $("tab-lab");

  function scale(p, t) {
    if (!t || t === 1) return p;
    const lg = Object.fromEntries(Object.entries(p).map(([k, v]) => [k, Math.log(Math.max(v, 1e-9)) / t]));
    const m = Math.max(...Object.values(lg));
    const ex = Object.fromEntries(Object.entries(lg).map(([k, v]) => [k, Math.exp(v - m)]));
    const z = Object.values(ex).reduce((a, b) => a + b, 0);
    return Object.fromEntries(Object.entries(ex).map(([k, v]) => [k, v / z]));
  }
  const cal = (d) => scale(d.probabilities, D.temperatures[d.question]);

  async function show() {
    if (!D) root().innerHTML = `<div class="col wide"><div class="card"><div class="empty">Loading the Lab…</div></div></div>`;
    await load();
  }
  async function load() {
    [D, DEC] = await Promise.all([api("/api/lab"), api("/api/lab/decisions")]);
    TH = { ...D.thresholds };
    render();
  }

  function render() {
    root().innerHTML = `
      <div class="col wide">
        <section class="card"><h2>What if… <small>drag a threshold and see which past decisions would change · nothing runs until you save</small></h2>
          <div class="whatif" id="whatif"></div>
          <div class="row"><span class="kbd" id="th-note"></span><span class="spacer"></span>
            <button class="btn" id="th-reset">Reset to defaults</button>
            <button class="btn primary" id="th-save">Save thresholds</button></div>
        </section>
        <div class="lab-grid">
          <section class="card"><h2>Calibration <small>make Laya's probabilities honest</small></h2><div id="cal"></div></section>
          <section class="card"><h2>Question editor <small>reword a question, score it on your labels</small></h2><div id="editor"></div></section>
        </div>
      </div>`;
    renderWhatIf();
    renderCal();
    renderEditor();
    $("th-save").onclick = saveThresholds;
    $("th-reset").onclick = async () => {
      D = await api("/api/lab/thresholds", { values: Object.fromEntries(Object.keys(TH).map((k) => [k, null])) });
      TH = { ...D.thresholds }; renderWhatIf();
    };
    if (!watching) {
      watching = true;
      onResize(root(), () => { if (root().hidden) return; renderWhatIf(); if (calRes) drawReliability(); });
    }
  }
  let watching = false;

  /* ---------- 2. what-if thresholds ---------- */
  function renderWhatIf() {
    const box = $("whatif");
    box.innerHTML = Object.entries(SLIDERS).map(([name, s]) => {
      const locked = D.env_locked.includes(name);
      return `<div class="wi" data-name="${name}">
        <div class="wi-head"><b>${s.title}</b><span class="kbd">${esc(s.what)}${D.temperatures[s.q] !== 1 ? ` · calibrated (T=${D.temperatures[s.q].toFixed(2)})` : ""}</span></div>
        <div class="wi-row"><input type="range" min="0" max="1" step="0.01" value="${TH[name]}" ${locked ? "disabled" : ""}>
          <span class="wi-val">${pct(TH[name])}</span>${locked ? `<span class="tag warn">set by env var</span>` : ""}</div>
        <div class="wi-strip"></div><div class="wi-impact kbd"></div><div class="wi-sweep"></div></div>`;
    }).join("");
    box.querySelectorAll(".wi").forEach((el) => {
      const name = el.dataset.name, input = el.querySelector("input");
      input.oninput = () => { TH[name] = +input.value; el.querySelector(".wi-val").textContent = pct(TH[name]); drawSlider(el, name); updateNote(); };
      drawSlider(el, name);
    });
    updateNote();
  }

  function drawSlider(el, name) {
    const s = SLIDERS[name], th = TH[name];
    const ds = DEC.filter((d) => d.question === s.q);
    const cats = [{ key: "yes", name: s.yesName, color: "var(--s1)" }, { key: "no", name: s.noName, color: "var(--s2)" },
      { key: "none", name: "unlabelled", color: "var(--faint)" }];
    const pts = ds.map((d) => {
      const p = cal(d), x = s.x(p);
      const cat = d.label == null ? "none" : s.yes(d, p) ? "yes" : "no";
      return { x, cat, tip: `<b>${esc(d.text || "(no text)")}</b><br>${s.q === "task" ? `Laya: ${top(p)} ` : ""}${pct(x)}`
        + (d.label != null ? ` · label: ${esc(optName(s.q, d.label))}` : " · unlabelled") };
    });
    if (!pts.length) { el.querySelector(".wi-strip").innerHTML = `<div class="empty">No ${QN[s.q].toLowerCase()} decisions logged yet.</div>`; }
    else stripPlot(el.querySelector(".wi-strip"), pts, cats, { threshold: th, thresholdLabel: "" });

    // Impact against the saved thresholds, using the same (calibrated) probabilities.
    const saved = D.thresholds[name];
    const above = (t) => pts.filter((p) => p.x >= t).length;
    const lab = pts.filter((p) => p.cat !== "none");
    const acc = (t) => lab.length ? lab.filter((p) => (p.x >= t) === (p.cat === "yes")).length / lab.length : null;
    let impact = `${above(th)} of ${pts.length} at or above`;
    if (Math.abs(th - saved) > 1e-9) impact += ` (saved ${pct(saved)}: ${above(saved)})`;
    if (name === "TASK_MIN_PROB" || name === "HARD_MIN_PROB") impact += " · " + routingImpact();
    if (lab.length) impact += ` · right on ${Math.round(acc(th) * lab.length)}/${lab.length} labelled`;
    el.querySelector(".wi-impact").textContent = impact;

    // Accuracy sweep: where would the threshold do best on your labels?
    const sweep = el.querySelector(".wi-sweep");
    if (lab.length >= 4) {
      const xs = Array.from({ length: 51 }, (_, i) => i / 50);
      const best = xs.reduce((a, b) => (acc(b) > acc(a) ? b : a));
      lineChart(sweep, [{ name: "accuracy", color: "var(--s1)", markers: false,
        points: xs.map((x) => ({ x, y: acc(x), tip: `threshold ${pct(x)} → ${pct(acc(x))} right` })) }],
        { x: [0, 1], y: [0, 1], xFmt: pct, yFmt: pct, height: 96, yTicks: 2, xTicks: 4, steps: true });
      sweep.insertAdjacentHTML("afterbegin", `<div class="sub-h">Accuracy on your labels by threshold · best ${pct(best)} (${pct(acc(best))})</div>`);
    } else sweep.innerHTML = `<div class="kbd">Label ${4 - lab.length} more to see the accuracy-by-threshold curve.</div>`;
  }

  // How many past runs would route differently (model / thinking / temperature) with TH vs saved.
  function routingImpact() {
    const runs = {};
    for (const d of DEC) if (d.run && (d.question === "task" || d.question === "hard")) (runs[d.run] ||= {})[d.question] = cal(d);
    const decide = (r, th) => {
      const task = top(r.task), sure = r.task[task] >= th.TASK_MIN_PROB, hard = r.hard.A >= th.HARD_MIN_PROB;
      return `${hard || (sure && task === "code") ? "big" : "small"}|${hard}|${sure ? task : "unsure"}`;
    };
    let changed = 0, n = 0;
    for (const r of Object.values(runs)) {
      if (!r.task || !r.hard) continue;
      n++;
      if (decide(r, TH) !== decide(r, D.thresholds)) changed++;
    }
    return `${changed} of ${n} past runs would route differently`;
  }

  function updateNote() {
    const dirty = Object.keys(TH).filter((k) => Math.abs(TH[k] - D.thresholds[k]) > 1e-9);
    $("th-note").textContent = dirty.length ? `Unsaved: ${dirty.map((k) => SLIDERS[k].title.toLowerCase()).join(", ")}` : "Saved thresholds are live in the pipeline.";
    $("th-save").disabled = !dirty.length;
  }

  async function saveThresholds() {
    D = await api("/api/lab/thresholds", { values: TH });
    TH = { ...D.thresholds };
    renderWhatIf();
  }

  /* ---------- 3. calibration ---------- */
  const qButtons = (cur, id) => `<div class="seg" id="${id}">${Object.keys(QN).map((q) =>
    `<button data-q="${q}" class="${q === cur ? "on" : ""}">${QN[q]} <span class="kbd">${D.counts[q]}</span></button>`).join("")}</div>`;

  function renderCal() {
    const t = D.temperatures;
    $("cal").innerHTML = `${qButtons(calQ, "cal-q")}
      <p class="help">Laya's probabilities aren't honest out of the box: sometimes too sure, sometimes too timid.
      Fitting one temperature <b>T</b> per question on your labelled examples fixes how sure it claims to be
      (T&nbsp;&gt;&nbsp;1 softens, T&nbsp;&lt;&nbsp;1 sharpens) without changing which option wins.
      Thresholds then mean what they say.</p>
      <div class="row"><span class="kbd">${QN[calQ]}: ${D.counts[calQ]} labelled · live T = ${t[calQ].toFixed(2)}${t[calQ] === 1 ? " (uncalibrated)" : ""}</span>
        <span class="spacer"></span>
        ${t[calQ] !== 1 ? `<button class="btn" id="cal-remove">Remove calibration</button>` : ""}
        <button class="btn primary" id="cal-fit" ${D.counts[calQ] < 8 ? "disabled" : ""}>Fit on ${D.counts[calQ]} examples</button></div>
      ${D.counts[calQ] < 8 ? `<div class="kbd" style="margin-top:6px">Needs at least 8 labelled examples — label runs in the Live tab's History or add test cases in the editor.</div>` : ""}
      <div id="cal-out"></div>`;
    $("cal-q").querySelectorAll("button").forEach((b) => b.onclick = () => { calQ = b.dataset.q; calRes = null; renderCal(); });
    $("cal-fit").onclick = fitCal;
    if ($("cal-remove")) $("cal-remove").onclick = async () => { D = await api("/api/lab/temperature", { question: calQ, T: null }); calRes = null; renderCal(); renderWhatIf(); };
    if (calRes && calRes.question === calQ) drawCal();
  }

  async function fitCal() {
    $("cal-fit").disabled = true; $("cal-fit").textContent = "Running Laya on your examples…";
    try { calRes = await api("/api/lab/calibrate", { question: calQ }); drawCal(); }
    catch (e) { $("cal-out").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
    finally { $("cal-fit").disabled = false; $("cal-fit").textContent = `Refit on ${D.counts[calQ]} examples`; }
  }

  function drawCal() {
    const r = calRes, b = r.before, a = r.after;
    $("cal-out").innerHTML = `
      <div class="stats" style="margin-top:12px">
        <div class="stat"><div class="v">${r.T.toFixed(2)}</div><div class="k">fitted T</div></div>
        <div class="stat"><div class="v">${pct(b.ece)} → ${pct(a.ece)}</div><div class="k">calibration error</div></div>
        <div class="stat"><div class="v">${b.log_loss.toFixed(2)} → ${a.log_loss.toFixed(2)}</div><div class="k">log loss</div></div>
        <div class="stat"><div class="v">${pct(a.accuracy)}</div><div class="k">accuracy (unchanged)</div></div>
      </div>
      <div class="sub-h">Reliability: how often Laya was right vs how sure it said it was · on the diagonal = honest</div>
      <div id="rel"></div>
      <div class="row"><span class="kbd">${r.n} examples · ${r.T > 1 ? "Laya was over-confident" : r.T < 1 ? "Laya was under-confident" : "already calibrated"}</span><span class="spacer"></span>
        <button class="btn primary" id="cal-apply">Use T = ${r.T.toFixed(2)} live</button></div>`;
    drawReliability();
    $("cal-apply").onclick = async () => { D = await api("/api/lab/temperature", { question: calQ, T: r.T }); renderCal(); renderWhatIf(); };
  }

  function drawReliability() {
    const box = $("rel"); if (!box) return;
    const pts = (bins) => bins.map((x) => ({ x: x.confidence, y: x.accuracy,
      tip: `${pct(x.lo)}–${pct(x.hi)} sure · ${x.n} examples<br>said ${pct(x.confidence)}, right ${pct(x.accuracy)}` }));
    lineChart(box, [
      { name: "before", color: "var(--s2)", points: pts(calRes.before.bins) },
      { name: "after", color: "var(--s1)", points: pts(calRes.after.bins) },
    ], { x: [0, 1], y: [0, 1], xFmt: pct, yFmt: pct, diagonal: true, height: 220 });
  }

  /* ---------- 1. question editor ---------- */
  function renderEditor() {
    const q = D.questions[edQ], edited = D.edited.includes(edQ);
    const cases = D.cases.filter((c) => c.question === edQ);
    $("editor").innerHTML = `${qButtons(edQ, "ed-q")}
      <label class="fld">Question Laya is asked ${edited ? `<span class="tag warn">edited</span>` : ""}
        <textarea id="ed-instr" rows="2">${esc(q.instructions)}</textarea></label>
      ${Object.entries(q.criteria).map(([k, v]) => `<label class="fld">Option <b>${esc(optName(edQ, k))}</b> <span class="kbd">${esc(k)}</span>
        <input data-k="${esc(k)}" value="${esc(v)}"></label>`).join("")}
      <div class="row"><span class="kbd">${D.counts[edQ]} labelled examples</span><span class="spacer"></span>
        ${edited ? `<button class="btn" id="ed-default">Restore default</button>` : ""}
        <button class="btn" id="ed-eval" ${D.counts[edQ] ? "" : "disabled"}>Score edit vs live</button>
        <button class="btn primary" id="ed-save">Make live</button></div>
      <div id="ed-out"></div>
      <div class="sub-h">Test cases for ${QN[edQ].toLowerCase()} <span class="kbd">(labelled runs from History count too)</span></div>
      <div class="cases">${cases.map((c) => `<div class="case"><span class="tag">${esc(optName(edQ, c.label))}</span>
        <span class="ctext" title="${esc(c.state)}">${esc(c.state)}</span><span class="kbd">${c.source}</span>
        <button class="x" data-id="${c.id}" title="Delete">×</button></div>`).join("") || `<div class="kbd">No test cases yet.</div>`}</div>
      <div class="row add-case"><input id="case-text" placeholder="A request to test, e.g. 'Explain recursion to a 5-year-old'">
        <select id="case-label">${Object.keys(q.criteria).map((k) => `<option value="${esc(k)}">${esc(optName(edQ, k))}</option>`).join("")}</select>
        <button class="btn" id="case-add">Add</button></div>`;
    $("ed-q").querySelectorAll("button").forEach((b) => b.onclick = () => { edQ = b.dataset.q; evalRes = null; renderEditor(); });
    const draft = () => ({ instructions: $("ed-instr").value.trim(),
      criteria: Object.fromEntries([...$("editor").querySelectorAll("input[data-k]")].map((i) => [i.dataset.k, i.value.trim()])) });
    $("ed-eval").onclick = async () => {
      $("ed-eval").disabled = true; $("ed-eval").textContent = "Scoring…";
      try { evalRes = await api("/api/lab/evaluate", { question: edQ, schema: draft() }); drawEval(); }
      catch (e) { $("ed-out").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
      finally { $("ed-eval").disabled = false; $("ed-eval").textContent = "Score edit vs live"; }
    };
    $("ed-save").onclick = async () => {
      if (D.temperatures[edQ] !== 1 && !confirm("Changing the wording removes this question's calibration (it was fitted to the old wording). Continue?")) return;
      D = await api("/api/lab/question", { question: edQ, schema: draft() }); evalRes = null; render();
    };
    if ($("ed-default")) $("ed-default").onclick = async () => { D = await api("/api/lab/question", { question: edQ, schema: null }); evalRes = null; render(); };
    $("editor").querySelectorAll(".case .x").forEach((b) => b.onclick = async () => {
      D = await api(`/api/lab/cases/${b.dataset.id}`, undefined, "DELETE"); renderEditor(); renderCal(); });
    $("case-add").onclick = async () => {
      const text = $("case-text").value.trim(); if (!text) return;
      D = await api("/api/lab/cases", { question: edQ, state: text, label: $("case-label").value });
      renderEditor(); renderCal();
    };
    if (evalRes) drawEval();
  }

  function drawEval() {
    const c = evalRes.current, e = evalRes.edited;
    const delta = e.accuracy - c.accuracy;
    const rows = c.items.map((it, i) => ({ it, ed: e.items[i] }));
    $("ed-out").innerHTML = `
      <div class="stats" style="margin-top:12px">
        <div class="stat"><div class="v">${pct(c.accuracy)}</div><div class="k">live wording</div></div>
        <div class="stat"><div class="v">${pct(e.accuracy)}</div><div class="k">your edit · ${delta > 0 ? "+" : ""}${Math.round(delta * 100)} pts</div></div>
        <div class="stat"><div class="v">${c.log_loss.toFixed(2)} → ${e.log_loss.toFixed(2)}</div><div class="k">log loss (lower = surer when right)</div></div>
      </div>
      <div class="hist-wrap"><table class="hist evaltab"><tr><th>Example</th><th>Label</th><th>Live</th><th>Edit</th></tr>
      ${rows.map(({ it, ed }) => {
        const txt = typeof it.state === "string" ? it.state
          : it.state.passage ? `${it.state.question} ⟶ ${it.state.passage}` : it.state.answer ? `${it.state.question} ⟶ ${it.state.answer}` : JSON.stringify(it.state);
        const cell = (x) => `<span class="${x.predicted === x.label ? "okt" : "badt"}">${x.predicted === x.label ? "✓" : "✗"} ${esc(optName(edQ, x.predicted))}</span> <span class="kbd">${pct(x.probabilities[x.predicted])}</span>`;
        return `<tr class="${it.predicted !== ed.predicted ? "changed" : ""}"><td class="q" title="${esc(txt)}">${esc(txt)}</td>
          <td>${esc(optName(edQ, it.label))}</td><td>${cell(it)}</td><td>${cell(ed)}</td></tr>`;
      }).join("")}</table></div>
      <div class="kbd">Highlighted rows change answer with your edit. Probabilities shown are calibrated with the live T.</div>`;
  }

  return { show };
})();
