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

let TASK = "multiclass";
let THR = 0.5;

const pct1 = (v) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—");
const num3 = (v) => (typeof v === "number" ? v.toFixed(3) : "—");

async function loadMeta() {
  let m;
  try {
    m = await (await fetch("/api/meta")).json();
  } catch (e) {
    $("device-pill").textContent = "server unreachable";
    return;
  }
  TASK = m.task || "multiclass";
  THR = Number(TASK === "binary" ? m.decision_threshold ?? m.abstain_threshold : m.abstain_threshold);
  $("device-pill").textContent = m.backbone + " · " + m.device;
  if (TASK === "binary") {
    $("idle-text").textContent =
      "The gate scores each photo for a dental issue that needs attention. Upload a photo to see the probability.";
  } else if (m.classes) {
    $("idle-text").textContent =
      "The model returns a probability for each of " + m.classes.length +
      " conditions. Upload a photo to see them.";
  }
  buildMetrics(m);
  buildCkpt(m);
  setAbstainCopy();
  buildFooter(m);
  // Checkpoint `notes` intentionally not shown — the footer disclaimer covers it.
}

function buildMetrics(m) {
  const met = m.metrics || {};
  const t = $("metrics-tbl").querySelector("tbody");
  const f = $("f1-bars");
  f.innerHTML = "";
  let mt, ci, n, rows;
  if (TASK === "binary") {
    mt = (met.test_at_threshold || {}).test || {};
    ci = (met.macro_f1_95ci_at_0_5 || {}).test;
    n = met.images && met.images.test;
    rows = [
      ["Accuracy", mt.acc],
      ["Balanced accuracy", mt.bal_acc],
      ["Macro F1", mt.macro_f1],
      ["Sensitivity", mt.sensitivity],
      ["Specificity", mt.specificity],
      ["AUROC", mt.auroc],
      ["Average precision", mt.ap],
      ["95% CI (macro F1 @ 0.5)", ci],
    ];
    $("f1-h").textContent = "Test sensitivity & specificity";
    [["Sensitivity", mt.sensitivity], ["Specificity", mt.specificity], ["AUROC", mt.auroc]].forEach(
      ([l, v]) => typeof v === "number" && f.appendChild(barRow(l, v, false))
    );
  } else {
    mt = met.test || {};
    ci = met.test_macro_f1_95ci;
    n = met.n_test;
    rows = [
      ["Accuracy", mt.acc],
      ["Balanced accuracy", mt.bal_acc],
      ["Macro F1", mt.macro_f1],
      ["95% CI (macro F1)", ci],
    ];
    (m.classes || []).forEach((c) => {
      const v = mt["f1_" + c.name];
      if (typeof v === "number") f.appendChild(barRow(c.display_name, v, false));
    });
  }
  t.innerHTML = rows
    .filter((r) => r[1] !== undefined && r[1] !== null)
    .map(
      (r) =>
        `<tr><th>${r[0]}</th><td class="num">${typeof r[1] === "number" ? r[1].toFixed(3) : r[1]}</td></tr>`
    )
    .join("");
  $("perf-h").textContent = "Test performance" + (n != null ? " (n=" + n.toLocaleString() + ")" : "");
}

function buildCkpt(m) {
  const rows = [
    ["Experiment", m.experiment],
    ["Task", TASK === "binary" ? "issue gate (binary)" : "condition classifier"],
    ["Classes", m.classes ? m.classes.length : "—"],
    ["Input", m.img_size + "×" + m.img_size + " px"],
    ["Temperature", Number(m.temperature).toFixed(3)],
    TASK === "binary" ? ["Flag p(issue) ≥", THR.toFixed(2)] : ["Abstain below", THR.toFixed(2)],
  ];
  if (TASK === "binary" && m.target_sensitivity != null)
    rows.push(["Target sensitivity", Number(m.target_sensitivity).toFixed(2)]);
  const mg = m.mouth_gate || {};
  rows.push(["Mouth gate", mg.enabled ? "on (YuNet + CLIP)" : "off" + (mg.error ? " — failed to load" : "")]);
  const dt = m.detector || {};
  rows.push(["Detector", dt.enabled
    ? dt.run + " · boxes ≥ " + Number(dt.conf).toFixed(2) + (dt.metrics && dt.metrics.mAP50 != null
      ? " · test mAP50 " + num3(dt.metrics.mAP50) : "")
    : "off" + (dt.error ? " — failed to load" : "")]);
  rows.push(["Created", m.created]);
  $("ckpt-tbl").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1] ?? "—"}</td></tr>`).join("");
}

function setAbstainCopy() {
  const thr = THR.toFixed(2);
  $("abstain-body").textContent =
    TASK === "binary"
      ? "p(issue) is at or above the decision threshold (" + thr +
        ") — the gate recommends a dental check for this photo."
      : "Confidence is below the abstention threshold (" + thr +
        "). The model is not confident enough to answer — a retake with better lighting and focus, " +
        "or a dental visit, is the safer next step.";
}

function buildFooter(m) {
  const met = m.metrics || {};
  const T = Number(m.temperature).toFixed(2);
  const thr = THR.toFixed(2);
  let text;
  if (TASK === "binary") {
    const t = (met.test_at_threshold || {}).test || {};
    const x = (met.test_at_threshold || {}).xsrc_test || {};
    const tot = met.images ? Object.values(met.images).reduce((a, b) => a + b, 0) : null;
    text =
      "Trained on " + (tot ? tot.toLocaleString() : "thousands of") +
      " photos from multiple public datasets (group-aware splits, class-balanced sampling, strong augmentation). " +
      "At the decision threshold: sensitivity " + pct1(t.sensitivity) + ", specificity " + pct1(t.specificity) +
      ", AUROC " + num3(t.auroc) + " (cross-source AUROC " + num3(x.auroc) + "). " +
      "Probabilities are temperature-scaled (T=" + T + "); photos with p(issue) at or above " + thr + " are flagged.";
  } else {
    const t = met.test || {};
    text =
      "Trained on 4,375 images from the Kaggle oral-image dataset (group-aware splits, class-balanced sampling, " +
      "strong augmentation). Test macro-F1 " + num3(t.macro_f1) + " " + (met.test_macro_f1_95ci || "") + " on " +
      (met.n_test != null ? met.n_test : "held-out") + " held-out photos. " +
      "Probabilities are temperature-scaled (T=" + T + "); predictions under the abstention threshold (" + thr +
      ") are flagged.";
  }
  const dt = m.detector || {};
  if (dt.enabled) {
    const dm = dt.metrics || {};
    text += " Flagged photos are passed to a YOLO detector (" + dt.run + ") that outlines calculus, caries, " +
      "hypodontia, mouth ulcers and tooth discoloration" +
      (dm.mAP50 != null ? " (test mAP50 " + num3(dm.mAP50) + ", mAP50-95 " + num3(dm.mAP50_95) + ")" : "") + ".";
  }
  $("disclaimer").textContent = text;
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
  $("rejected").classList.add("hidden");
  $("error").classList.add("hidden");
  setMouthBox(null);
  classify(file, url);
}

function clearFile() {
  currentFile = null;
  fileInput.value = "";
  previewCard.classList.add("hidden");
  $("result").classList.add("hidden");
  $("rejected").classList.add("hidden");
  $("error").classList.add("hidden");
  $("idle").classList.remove("hidden");
  setMouthBox(null);
}

/* ---------- mouth gate ---------- */

// box: four [x, y] corners as fractions of the image (the crop can be rotated).
function setMouthBox(box) {
  const svg = $("mouth-box");
  if (!box) {
    svg.classList.add("hidden");
    return;
  }
  svg.querySelector("polygon").setAttribute("points", box.map((p) => p.join(",")).join(" "));
  svg.classList.remove("hidden");
}

const SCENE_LABEL = { intraoral: "Mouth close-up", face: "Face / person", other: "Something else" };
const REJECT_TITLE = {
  no_mouth: "No mouth found",
  mouth_not_found: "Couldn't locate the mouth",
  mouth_too_small: "Mouth too small in the frame",
  mouth_closed: "Mouth looks closed",
};

function renderRejected(r, wallMs, file, thumbUrl) {
  const g = r.gate || {};
  previewImg.src = thumbUrl;
  setMouthBox(g.box);
  $("rej-title").textContent = REJECT_TITLE[g.reason] || "No mouth to classify";
  $("rej-msg").textContent = g.message || "";
  const bars = $("rej-bars");
  bars.innerHTML = "";
  const scene = g.scene || {};
  Object.keys(scene)
    .sort((a, b) => scene[b] - scene[a])
    .forEach((k, i) => bars.appendChild(barRow(SCENE_LABEL[k] || k, scene[k], i === 0)));
  if (typeof g.p_open === "number") bars.appendChild(barRow("Mouth open", g.p_open, false));
  $("rej-latency").textContent =
    (g.gate_ms != null ? `gate ${g.gate_ms} ms · ` : "") + `round trip ${wallMs} ms · ${file.name}`;
  $("idle").classList.add("hidden");
  $("error").classList.add("hidden");
  $("result").classList.add("hidden");
  $("rejected").classList.remove("hidden");
}

function renderGate(g) {
  const row = $("gate-row"), crop = $("gate-crop");
  if (!g) {
    row.classList.add("hidden");
    setMouthBox(null);
    return;
  }
  $("gate-msg").textContent = g.message || "";
  if (g.crop_image) {
    crop.querySelector("img").src = g.crop_image;
    crop.classList.remove("hidden");
  } else {
    crop.classList.add("hidden");
  }
  setMouthBox(g.box);
  row.classList.remove("hidden");
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
  if (r.rejected) {
    renderRejected(r, wallMs, file, thumbUrl);
    if (file.name !== "(from history)") addHistory(r, thumbUrl);
    return;
  }
  previewImg.src = thumbUrl;
  $("rejected").classList.add("hidden");
  renderGate(r.gate);
  $("pred-name").textContent = r.display_name;
  $("pred-conf").textContent = fmtPct(r.confidence);
  if (r.task === "binary") {
    $("subline").textContent =
      "p(issue) " + fmtPct(r.p_issue) + " · flagged at ≥ " + fmtPct(r.threshold);
  } else {
    $("subline").textContent =
      "Runner-up: " + r.top2.map((t) => `${t.display_name} ${fmtPct(t.prob)}`).join("  ·  ");
  }
  $("latency").textContent =
    (r.gate && r.gate.gate_ms != null ? `gate ${r.gate.gate_ms} ms · ` : "") +
    `model ${r.inference_ms} ms · round trip ${wallMs} ms · ${file.name}`;
  $("abstain").classList.toggle("hidden", !r.abstain);

  const bars = $("bars");
  bars.innerHTML = "";
  const names = Object.keys(r.probs).sort((a, b) => r.probs[b] - r.probs[a]);
  names.forEach((name, i) => bars.appendChild(barRow(displayName(name), r.probs[name], i === 0)));

  renderDetection(r);
  // The detector's image is a large data URL; keep it out of the JSON viewer.
  const shown = r.detection ? { ...r, detection: { ...r.detection, image: "(data URL omitted)" } } : r;
  $("raw-json").textContent = JSON.stringify(shown, null, 2);
  $("idle").classList.add("hidden");
  $("error").classList.add("hidden");
  $("result").classList.remove("hidden");

  if (file.name !== "(from history)") addHistory(r, thumbUrl);
}

/* ---------- stage-2 detector ---------- */

// Okabe-Ito, the same colours as the detector notebook's figures.
const DET_COLORS = {
  Calculus: "#E69F00",
  Caries: "#D55E00",
  Hypodontia: "#0072B2",
  Mouth_Ulcer: "#CC79A7",
  Tooth_Discoloration: "#009E73",
};
const detColor = (c) => DET_COLORS[c] || "#475569";
let detCurrent = null;

function renderDetection(r) {
  const card = $("det-card");
  detCurrent = r.detection || null;
  if (!detCurrent) {
    card.classList.add("hidden");
    return;
  }
  const d = detCurrent;
  const weak = d.boxes.length - d.n_confident;
  $("det-img").src = d.image;
  // Nothing confident: the best guesses are always drawn (no toggle), so a flagged photo is outlined.
  $("det-weak-wrap").classList.toggle("hidden", weak === 0 || d.n_confident === 0);
  $("det-weak").checked = false;
  drawDetection();
  card.classList.remove("hidden");
}

function drawDetection() {
  const d = detCurrent;
  if (!d) return;
  const showWeak = $("det-weak").checked || d.n_confident === 0;
  const boxes = d.boxes.filter((b) => b.confident || showWeak);
  const layer = $("det-boxes");
  layer.innerHTML = "";
  // Weakest first so the strongest boxes (and their tags) are drawn on top.
  [...boxes].reverse().forEach((b) => {
    const [x1, y1, x2, y2] = b.box;
    const el = document.createElement("div");
    el.className = "det-box" + (b.confident ? "" : " weak") + (y1 < 0.06 ? " top-edge" : "") +
      (x2 > 0.7 ? " right-edge" : "");
    el.style.setProperty("--c", detColor(b.class));
    Object.assign(el.style, {
      left: x1 * 100 + "%", top: y1 * 100 + "%",
      width: (x2 - x1) * 100 + "%", height: (y2 - y1) * 100 + "%",
    });
    el.innerHTML = `<span class="det-tag">${b.display_name} ${Math.round(b.conf * 100)}%</span>`;
    layer.appendChild(el);
  });

  // One chip per condition: count and strongest confidence.
  const byClass = {};
  boxes.forEach((b) => {
    const k = byClass[b.class] || (byClass[b.class] = { n: 0, max: 0, confident: false, name: b.display_name });
    k.n += 1;
    k.max = Math.max(k.max, b.conf);
    k.confident = k.confident || b.confident;
  });
  $("det-list").innerHTML = Object.entries(byClass)
    .sort((a, b) => b[1].max - a[1].max)
    .map(([c, k]) =>
      `<span class="det-chip${k.confident ? "" : " weak"}" style="--c:${detColor(c)}"><i></i>${k.name}` +
      `${k.n > 1 ? " ×" + k.n : ""} <span class="mono">${fmtPct(k.max)}</span></span>`)
    .join("");

  const note = $("det-note");
  const thr = Math.round(d.conf * 100) + "%";
  note.classList.toggle("warn", d.n_confident === 0);
  if (d.n_confident === 0 && d.boxes.length === 0) {
    note.textContent = "The photo was flagged, but the detector found nothing to outline — the issue may be " +
      "outside its five conditions (e.g. gum disease), too subtle at this resolution, or the flag a false alarm.";
  } else if (d.n_confident === 0) {
    note.textContent = "No finding reached the detector's " + thr + " threshold, so the dashed boxes are its " +
      "best guesses for where the flagged issue is — low confidence, so treat them as hints. On test photos, " +
      "most flagged images without a confident box were healthy (classifier false alarms).";
  } else {
    note.textContent = d.n_confident + (d.n_confident === 1 ? " finding" : " findings") + " at ≥ " + thr +
      " confidence" + (showWeak && boxes.length > d.n_confident ? "; dashed boxes are weaker candidates" : "") +
      ". Boxes show where to look, not a diagnosis. · detector " + d.detector_ms + " ms";
  }
}

$("det-weak").addEventListener("change", drawDetection);

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
    const rej = h.r.rejected;
    const name = rej ? REJECT_TITLE[(h.r.gate || {}).reason] || "Not classified" : displayName(h.r.class_name);
    const crop = !rej && h.r.gate && h.r.gate.route === "face_crop" ? " · mouth cropped" : "";
    const nd = !rej && h.r.detection ? h.r.detection.n_confident : null;
    const found = nd == null ? "" : " · " + (nd ? nd + (nd === 1 ? " finding" : " findings")
      : h.r.detection.boxes.length ? "best guess only" : "no box");
    div.innerHTML = `
      <img src="${h.thumbUrl}" alt="">
      <div class="hist-main">
        <div class="hist-name">${name}</div>
        <div class="hist-sub">${h.when.toLocaleTimeString()}${crop}${found}</div>
      </div>
      ${rej
        ? `<span class="hist-conf hist-reject">skipped</span>`
        : `<span class="hist-conf ${h.r.abstain ? "hist-abstain" : ""}">${fmtPct(h.r.confidence)}${h.r.abstain ? " ⚠" : ""}</span>`}`;
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
