/* Demo front-end for the oral condition classifier. */
"use strict";

const $ = (id) => document.getElementById(id);

const drop = $("drop"), fileInput = $("file");
const previewCard = $("preview-card"), previewImg = $("preview");
let currentFile = null;
let history = [];

const fmtPct = (p) => (p * 100).toFixed(1) + "%";
const fmtBytes = (n) =>
  n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(0) + " KB" : (n / 1048576).toFixed(1) + " MB";

/* ---------- meta / model strip ---------- */

async function loadMeta() {
  let m;
  try {
    m = await (await fetch("/api/meta")).json();
  } catch (e) {
    $("device-pill").textContent = "server unreachable";
    return;
  }
  $("device-pill").textContent = m.backbone + " · " + m.device;
  $("abstain-thresh").textContent = Number(m.abstain_threshold).toFixed(2);

  const t = $("metrics-tbl").querySelector("tbody");
  const mt = (m.metrics && m.metrics.test) || {};
  const rows = [
    ["Accuracy", mt.acc],
    ["Balanced accuracy", mt.bal_acc],
    ["Macro F1", mt.macro_f1],
    ["95% CI (macro F1)", m.metrics && m.metrics.test_macro_f1_95ci],
  ];
  t.innerHTML = rows
    .filter((r) => r[1] !== undefined && r[1] !== null)
    .map(
      (r) =>
        `<tr><th>${r[0]}</th><td class="num">${typeof r[1] === "number" ? r[1].toFixed(3) : r[1]}</td></tr>`
    )
    .join("");

  if (m.classes) {
    const f = $("f1-bars");
    f.innerHTML = "";
    m.classes.forEach((c) => {
      const v = mt["f1_" + c.name];
      if (v === undefined) return;
      f.appendChild(barRow(c.display_name, v, false));
    });
  }

  const k = $("ckpt-tbl");
  k.innerHTML = [
    ["Experiment", m.experiment],
    ["Classes", m.classes ? m.classes.length : "—"],
    ["Input", m.img_size + "×" + m.img_size + " px"],
    ["Temperature", Number(m.temperature).toFixed(3)],
    ["Abstain below", Number(m.abstain_threshold).toFixed(2)],
    ["Created", m.created],
  ]
    .map((r) => `<tr><td>${r[0]}</td><td>${r[1] ?? "—"}</td></tr>`)
    .join("");

  // Checkpoint `notes` intentionally not shown — the footer disclaimer covers it.
}

/* ---------- bars ---------- */

function barRow(label, p, isTop) {
  const row = document.createElement("div");
  row.className = "bar-row " + (isTop ? "top" : "other");
  row.innerHTML = `
    <span class="bar-label">${label}</span>
    <span class="bar-track"><span class="bar-fill"></span></span>
    <span class="bar-val">${fmtPct(p)}</span>`;
  requestAnimationFrame(() => (row.querySelector(".bar-fill").style.width = Math.max(0.5, p * 100) + "%"));
  return row;
}

/* ---------- upload ---------- */

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("That file is not an image.");
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  $("file-name").textContent = file.name;
  $("file-size").textContent = fmtBytes(file.size);
  previewCard.classList.remove("hidden");
  $("idle").classList.add("hidden");
  $("result").classList.add("hidden");
  $("error").classList.add("hidden");
  classify(file, url);
}

function clearFile() {
  currentFile = null;
  fileInput.value = "";
  previewCard.classList.add("hidden");
  $("result").classList.add("hidden");
  $("error").classList.add("hidden");
  $("idle").classList.remove("hidden");
}

async function classify(file, thumbUrl) {
  $("loading").classList.remove("hidden");
  const body = new FormData();
  body.append("file", file);
  let resp, data;
  const t0 = performance.now();
  try {
    resp = await fetch("/api/predict", { method: "POST", body });
    data = await resp.json();
  } catch (e) {
    finishLoading();
    showError("Lost contact with the server. Is app.py still running?");
    return;
  }
  finishLoading();
  if (!resp.ok) {
    showError(data.detail || `Server returned ${resp.status}.`);
    return;
  }
  renderResult(data, Math.round(performance.now() - t0), file, thumbUrl);
}

function finishLoading() {
  $("loading").classList.add("hidden");
}

function showError(msg) {
  $("error-msg").textContent = msg;
  $("result").classList.add("hidden");
  $("idle").classList.add("hidden");
  $("error").classList.remove("hidden");
}

/* ---------- result ---------- */

function renderResult(r, wallMs, file, thumbUrl) {
  $("pred-name").textContent = r.display_name;
  $("pred-conf").textContent = fmtPct(r.confidence);
  $("top2").textContent =
    r.top2.map((t) => `${t.display_name} ${fmtPct(t.prob)}`).join("  ·  ");
  $("latency").textContent = `model ${r.inference_ms} ms · round trip ${wallMs} ms · ${file.name}`;
  $("abstain").classList.toggle("hidden", !r.abstain);

  const bars = $("bars");
  bars.innerHTML = "";
  const names = Object.keys(r.probs).sort((a, b) => r.probs[b] - r.probs[a]);
  names.forEach((name, i) => bars.appendChild(barRow(displayName(name), r.probs[name], i === 0)));

  $("raw-json").textContent = JSON.stringify(r, null, 2);
  $("idle").classList.add("hidden");
  $("error").classList.add("hidden");
  $("result").classList.remove("hidden");

  addHistory(r, thumbUrl);
}

let DISPLAY = {};
async function loadDisplay() {
  try {
    const m = await (await fetch("/api/meta")).json();
    (m.classes || []).forEach((c) => (DISPLAY[c.name] = c.display_name));
  } catch (e) { /* footer strip will show defaults */ }
}
const displayName = (n) => DISPLAY[n] || n.replace(/_/g, " ");

/* ---------- history (page-local, not stored) ---------- */

function addHistory(r, thumbUrl) {
  history.unshift({ r, thumbUrl, when: new Date() });
  if (history.length > 8) history.pop();
  const box = $("history");
  box.innerHTML = "";
  history.forEach((h) => {
    const div = document.createElement("div");
    div.className = "hist-item";
    div.title = "Click to show this result again";
    div.innerHTML = `
      <img src="${h.thumbUrl}" alt="">
      <div class="hist-main">
        <div class="hist-name">${displayName(h.r.class_name)}</div>
        <div class="hist-sub">${h.when.toLocaleTimeString()}</div>
      </div>
      <span class="hist-conf ${h.r.abstain ? "hist-abstain" : ""}">${fmtPct(h.r.confidence)}${h.r.abstain ? " ⚠" : ""}</span>`;
    div.addEventListener("click", () =>
      renderResult(h.r, 0, { name: "(from history)" }, h.thumbUrl));
    box.appendChild(div);
  });
}

/* ---------- wiring ---------- */

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
$("again").addEventListener("click", () => fileInput.click());
$("clear").addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) setFile(f);
});

document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

loadDisplay().then(loadMeta);
