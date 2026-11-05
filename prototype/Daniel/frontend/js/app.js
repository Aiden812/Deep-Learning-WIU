// Gigi: patient flow. Photo -> result -> cost -> clinics & care.
// Everything lives in memory for this tab only; nothing is persisted.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = (id, cls = "ic") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${id}"/></svg>`;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const MAX_MB = 15;
const FALLBACK_EMERGENCY = [
  "Swelling in your face, jaw or neck",
  "Bleeding from the mouth that won't stop",
  "A tooth knocked out or broken in an accident",
  "Fever with a painful, swollen mouth",
  "Difficulty breathing, swallowing or opening your mouth",
];
const IF_WORSE_LABEL = {
  caries: "Only if the decay has reached the nerve",
  hypodontia: "If you decide to replace a missing tooth",
};

const state = {
  meta: null,
  step: 1,
  maxStep: 1,
  file: null,
  url: null,
  screen: null,
  condition: null,
  abstain: false,
  profile: { residency: "citizen", tier: "none", age: "under60" },
  estimate: null,
  coords: null,
};

/* ---------------- API ---------------- */
async function request(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Gigi's server can't be reached. Check your connection and try again.");
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error(body?.detail || `Something went wrong (error ${res.status}). Please try again.`);
  return body;
}
const postJSON = (path, data) => request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

/* ---------------- Helpers ---------------- */
const content = () => state.meta?.content;
const info = (name) => content()?.conditions?.[name];
const label = (name) => info(name)?.label ?? name;
const pct = (p) => `${Math.round(p * 100)}%`;
const money = (n) => `S$${Number(n).toLocaleString("en-SG", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const range = (a, b) => (a === b ? money(a) : `${money(a)} – ${money(b).replace("S$", "")}`);

function confidenceWords(p, threshold) {
  if (p < threshold) return "Unsure";
  if (p >= 0.85) return "Fairly confident";
  if (p >= 0.65) return "Moderately confident";
  return "Low confidence";
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 4200);
}

function busy(btn, on) {
  btn.classList.toggle("is-busy", on);
  btn.setAttribute("aria-busy", String(on));
  btn.disabled = on;
}

/* ---------------- Steps ---------------- */
function goTo(n, { focus = true } = {}) {
  n = Math.max(1, Math.min(n, state.maxStep));
  state.step = n;
  $$(".panel").forEach((p) => {
    const on = Number(p.dataset.step) === n;
    p.hidden = !on;
    p.classList.toggle("is-entering", on && !reduceMotion);
  });
  $$(".stepper li").forEach((li) => {
    const s = Number(li.dataset.stepNav);
    li.dataset.state = s === n ? "current" : s <= state.maxStep ? "done" : "todo";
    const b = $("button", li);
    b.disabled = s > state.maxStep || s === n;
    if (s === n) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
  });
  if (n === 3) loadEstimate();
  if (focus) {
    const host = $("#check");
    const top = host.getBoundingClientRect().top;
    if (top < 0 || top > innerHeight * 0.4) host.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    const panel = $(`.panel[data-step="${n}"]`);
    const h = $$(".panel-h", panel).find((el) => el.offsetParent !== null);
    h?.focus({ preventScroll: true });
  }
}

function unlock(n) { state.maxStep = Math.max(state.maxStep, n); }

/* ---------------- Step 1: photo ---------------- */
function renderEmergency() {
  const signs = content()?.emergency_signs ?? FALLBACK_EMERGENCY;
  $("#emergency-list").innerHTML = signs
    .map((s, i) => `<label><input type="checkbox" name="emergency" value="${i}">${esc(s)}</label>`)
    .join("");
}

function onEmergencyChange() {
  const any = $$("#emergency-list input").some((i) => i.checked);
  $("#emergency-stop").hidden = !any;
  $("#photo-area").hidden = any;
}

function setFile(file) {
  const err = $("#s1-error");
  err.hidden = true;
  const looksImage = file && (file.type.startsWith("image/") || /\.(heic|heif)$/i.test(file.name));
  if (!looksImage) {
    err.textContent = "That file isn't a photo. Choose a JPG, PNG or HEIC image.";
    err.hidden = false;
    return;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    err.textContent = `That photo is larger than ${MAX_MB} MB. Try a smaller one, or take a new photo.`;
    err.hidden = false;
    return;
  }
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file;
  state.url = URL.createObjectURL(file);
  const img = $("#preview-img");
  img.onerror = () => { img.removeAttribute("src"); img.alt = "Preview not available for this file type"; };
  img.src = state.url;
  $("#preview-name").textContent = file.name || "Photo";
  $("#preview-size").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("#preview").hidden = false;
  $("#drop").hidden = true;
  $(".drop-or").hidden = true;
  updateAnalyse();
}

function clearFile() {
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = null;
  state.url = null;
  $("#file").value = "";
  $("#camera").value = "";
  $("#preview").hidden = true;
  $("#drop").hidden = false;
  $(".drop-or").hidden = false;
  updateAnalyse();
}

function updateAnalyse() {
  const ok = !!state.file && $("#consent").checked;
  $("#analyse").disabled = !ok;
  $("#analyse-why").textContent = !state.file
    ? "Add a photo and tick the consent box to continue."
    : !$("#consent").checked
      ? "Tick the consent box to continue."
      : "Takes a few seconds.";
}

async function demoPhoto() {
  // A simple drawn stand-in so the flow can be demonstrated without a real photo.
  const c = document.createElement("canvas");
  c.width = 640; c.height = 480;
  const g = c.getContext("2d");
  g.fillStyle = "#6b2730"; g.fillRect(0, 0, 640, 480);
  g.fillStyle = "#d9737c"; g.beginPath(); g.ellipse(320, 170, 300, 110, 0, 0, Math.PI * 2); g.fill();
  g.beginPath(); g.ellipse(320, 330, 300, 100, 0, 0, Math.PI * 2); g.fill();
  for (let i = 0; i < 8; i++) {
    const x = 50 + i * 70;
    g.fillStyle = i === 3 ? "#d8c38f" : "#f3eee3";
    g.beginPath(); g.roundRect(x, 150, 60, 95, 14); g.fill();
    g.fillStyle = "#efe8da";
    g.beginPath(); g.roundRect(x + 4, 255, 58, 85, 14); g.fill();
  }
  const d = g.getImageData(0, 0, 640, 480);
  for (let i = 0; i < d.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 34;
    d.data[i] += n; d.data[i + 1] += n; d.data[i + 2] += n;
  }
  g.putImageData(d, 0, 0);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
  setFile(new File([blob], "demo-photo.jpg", { type: "image/jpeg" }));
  $("#consent").focus();
}

async function analyse() {
  unlock(2);
  state.maxStep = 2; // a new screening invalidates later steps
  state.estimate = null;
  $("#agent-out").innerHTML = "";
  $("#print").hidden = true;
  goTo(2);
  $("#s2-loading").hidden = false;
  $("#s2-retake").hidden = true;
  $("#s2-result").hidden = true;
  $("#s2-error").hidden = true;

  const fd = new FormData();
  fd.append("file", state.file);
  const scenario = $("#scenario").value;
  if (scenario) fd.append("scenario", scenario);
  try {
    const [res] = await Promise.all([
      request("/api/screen", { method: "POST", body: fd }),
      new Promise((r) => setTimeout(r, reduceMotion ? 0 : 450)), // avoid a jarring flash
    ]);
    state.screen = res;
    renderResult(res);
  } catch (e) {
    $("#s2-loading").hidden = true;
    $("#s2-error-msg").textContent = e.message;
    $("#s2-error").hidden = false;
  }
}

/* ---------------- Step 2: result ---------------- */
function renderResult({ quality, result }) {
  $("#s2-loading").hidden = true;

  if (!quality.ok || !result) {
    $("#retake-issues").innerHTML = quality.issues
      .map((i) => `<li class="${i.severity}">${icon("alert")}<span>${esc(i.message)}</span></li>`)
      .join("");
    $("#s2-retake").hidden = false;
    goTo(2);
    return;
  }

  const c = info(result.top);
  state.condition = result.top;
  state.abstain = result.abstain;
  unlock(4);

  const threshold = result.abstain_threshold ?? 0.5;
  const words = confidenceWords(result.confidence, threshold);
  const simulated = result.mode === "demo" ? ` <span class="tag tag-plain tag-sample">Simulated result</span>` : "";
  $("#r-kicker").innerHTML = (result.abstain
    ? `Gigi can't tell for sure. The closest match is`
    : `${words} (${pct(result.confidence)}). This photo looks most like`) + simulated;
  $("#s2-h").textContent = c.label;
  $("#r-short").textContent = c.short;

  const urgencyKey = result.abstain ? "unsure" : c.urgency;
  const u = content().urgency[urgencyKey];
  const tag = $("#r-tag");
  tag.className = `tag tag-${urgencyKey}`;
  tag.textContent = u.label;
  $("#r-advice").textContent = u.advice;
  $("#r-abstain").hidden = !result.abstain;

  const rows = [
    ["What it is", c.what],
    ["Why it happens", c.why],
    ["If it's left alone", c.if_left],
    ["What a dentist usually does", c.dentist],
  ];
  $("#r-explain").innerHTML = rows.map(([t, d]) => `<div><dt>${t}</dt><dd>${esc(d)}</dd></div>`).join("");
  $("#r-home").innerHTML = c.home.map((h) => `<li>${esc(h)}</li>`).join("");

  $("#r-photo").src = state.url || "";
  const warns = quality.issues.filter((i) => i.severity === "warn");
  const cap = $("#r-quality");
  cap.classList.toggle("warn", warns.length > 0);
  cap.innerHTML = warns.length
    ? `${icon("alert")}<span>${esc(warns.map((w) => w.message).join(" "))} The result may be less reliable.</span>`
    : `${icon("check")}<span>Photo quality looks fine.</span>`;

  $("#r-threshold").textContent = pct(threshold);
  const spread = $("#r-spread");
  spread.innerHTML = result.ranked
    .map((r) => `<li><span>${esc(label(r.name))}</span><span class="num">${pct(r.prob)}</span><span class="bar is-pre" aria-hidden="true"><i style="--p:${r.prob.toFixed(3)}"></i></span></li>`)
    .join("");
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$(".bar", spread).forEach((b, i) => setTimeout(() => b.classList.remove("is-pre"), reduceMotion ? 0 : i * 45));
  }));

  $("#s2-result").hidden = false;
  goTo(2);
}

/* ---------------- Step 3: cost ---------------- */
function readProfile() {
  const f = $("#profile");
  const v = (n) => f.querySelector(`input[name="${n}"]:checked`)?.value;
  state.profile = { residency: v("residency"), tier: v("tier"), age: v("age") };
  const tierField = $("#tier-field");
  const citizen = state.profile.residency === "citizen";
  tierField.disabled = !citizen;
  tierField.querySelector("legend").textContent = citizen ? "CHAS card" : "CHAS card (Singapore Citizens only)";
}

let estimateSeq = 0;
async function loadEstimate() {
  if (!state.condition) return;
  readProfile();
  const seq = ++estimateSeq;
  const out = $("#cost-out");
  out.setAttribute("aria-busy", "true");
  out.style.opacity = state.estimate ? "0.55" : "";
  try {
    const est = await postJSON("/api/estimate", { condition: state.condition, ...state.profile });
    if (seq !== estimateSeq) return;
    state.estimate = est;
    renderCost(est);
  } catch (e) {
    if (seq !== estimateSeq) return;
    out.innerHTML = `<div class="alert alert-danger" role="alert">${icon("alert")}<div><p class="alert-h">Couldn't work out the cost.</p><p>${esc(e.message)}</p></div></div>`;
  } finally {
    if (seq === estimateSeq) { out.removeAttribute("aria-busy"); out.style.opacity = ""; }
  }
}

function renderCost(est) {
  const srcKeys = Object.keys(est.sources);
  const ref = (keys) => keys.map((k) => {
    const n = srcKeys.indexOf(k) + 1;
    return `<sup><a href="#src-${n}" aria-label="Source ${n}">${n}</a></sup>`;
  }).join("");
  const all = [...est.likely, ...est.if_worse];
  const showMedi = all.some((l) => l.medisave > 0);
  const chasEligible = est.profile.residency === "citizen" && !["none", "green"].includes(est.profile.tier);

  const chasCell = (l) => {
    if (l.subsidy_unknown) return `<span class="unk">Ask clinic*</span>`;
    if (l.subsidy > 0) return `<span class="neg">−${money(l.subsidy)}</span>`;
    return `<span class="muted">${chasEligible ? "Not covered" : "None"}</span>`;
  };

  const table = (lines, caption, sub) => `
    <div class="ledger-block">
      ${sub ? `<p class="ledger-sub">${esc(sub)}</p>` : ""}
      <table class="ledger">
        <caption>${esc(caption)}</caption>
        <thead><tr><th scope="col">Treatment</th><th scope="col">Typical fee</th><th scope="col">CHAS subsidy</th>${showMedi ? `<th scope="col">Flexi-MediSave</th>` : ""}<th scope="col">You might pay</th></tr></thead>
        <tbody>${lines.map((l) => `
          <tr>
            <td><span class="t-name">${esc(l.name)}</span><span class="t-plain">${esc(l.plain)}</span></td>
            <td class="num" data-label="Typical fee"><span>${range(l.fee_min, l.fee_max)}${ref([l.sources[0]])}</span></td>
            <td class="num" data-label="CHAS subsidy"><span>${chasCell(l)}${l.subsidy > 0 ? ref(l.sources.filter((s) => s.startsWith("chas"))) : ""}</span></td>
            ${showMedi ? `<td class="num" data-label="Flexi-MediSave"><span>${l.medisave > 0 ? `<span class="neg">−${money(l.medisave)}</span>${ref(l.sources.filter((s) => s.startsWith("cpf")))}` : `<span class="muted">None</span>`}</span></td>` : ""}
            <td class="num pay" data-label="You might pay"><span>${range(l.pay_min, l.pay_max)}</span></td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>`;

  const t = est.likely_total;
  const saved = Math.max(0, t.fee_max - t.pay_max);
  const basis = `${icon("info")}<span>Based on the screening result: <strong>${esc(label(est.condition))}</strong>${state.abstain ? " (uncertain, so treat this as a rough guide)" : ""}. A dentist decides the actual treatment.</span>`;

  let html = `<p class="cost-basis">${basis}</p>`;
  if (est.likely.length) {
    html += table(est.likely, "Most likely treatment");
    html += `
      <div class="total">
        <p class="total-label">You might pay<small>Most likely treatment, before GST</small></p>
        <p class="total-figure">${range(t.pay_min, t.pay_max)}</p>
        <p class="total-save"><span>Without subsidies: <span class="num">${range(t.fee_min, t.fee_max)}</span></span>${saved > 0 ? `<span>Subsidies could save you up to <strong class="num">${money(saved)}</strong></span>` : ""}</p>
      </div>`;
  } else {
    const msg = est.condition === "mouth_ulcer"
      ? "Most mouth ulcers heal on their own. If you need to see a dentist or doctor, a consultation fee applies. It varies by clinic and isn't in our sourced fee table."
      : "A dentist would first check this in person, usually with an X-ray. The consultation fee varies by clinic and isn't in our sourced fee table.";
    html += `<div class="empty-cost"><strong>No treatment fee to estimate yet</strong>${esc(msg)}</div>`;
  }
  if (est.if_worse.length) html += table(est.if_worse, IF_WORSE_LABEL[est.condition] ?? "If it has progressed", "Only if needed. A dentist will tell you.");

  html += `<ul class="notes">${est.notes.map((n) => `<li>${esc(n.startsWith("CHAS subsidies for this") ? "* " + n : n)}</li>`).join("")}</ul>`;
  html += `<p class="binding">${icon("info")}Non-binding estimate. Your clinic will confirm the actual fee.</p>`;
  html += `<div class="sources"><p class="sources-h">Sources</p><ol>${srcKeys.map((k, i) => {
    const s = est.sources[k];
    return `<li id="src-${i + 1}"><span>${i + 1}</span><span><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>, checked ${new Date(s.checked).toLocaleDateString("en-SG", { day: "numeric", month: "short", year: "numeric" })}</span></li>`;
  }).join("")}</ol></div>`;

  $("#cost-out").innerHTML = html;
}

/* ---------------- Step 4: agent ---------------- */
async function loadAreas() {
  const sel = $("#area");
  sel.innerHTML = `<option value="" disabled selected>Choose your town</option>`;
  try {
    const areas = await request("/api/areas");
    sel.insertAdjacentHTML("beforeend", areas.map((a) => `<option>${esc(a)}</option>`).join(""));
  } catch { /* the agent call will surface the connection problem */ }
}

function locate() {
  const msg = $("#locate-msg");
  const btn = $("#locate");
  if (!("geolocation" in navigator)) { msg.textContent = "Your browser can't share location. Choose your town instead."; return; }
  busy(btn, true);
  msg.textContent = "Asking your browser for your location…";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      busy(btn, false);
      state.coords = { lat: pos.coords.latitude, lng: pos.coords.longitude };
      const sel = $("#area");
      let opt = sel.querySelector('option[value="__here"]');
      if (!opt) { opt = new Option("My current location", "__here"); sel.insertBefore(opt, sel.options[1]); }
      sel.value = "__here";
      msg.textContent = "Using your current location. It's only used to sort clinics by distance.";
    },
    () => {
      busy(btn, false);
      msg.textContent = "We couldn't get your location. Choose your town instead.";
    },
    { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 },
  );
}

async function runAgent(ev) {
  ev.preventDefault();
  const out = $("#agent-out");
  const area = $("#area").value;
  if (!area) {
    out.innerHTML = `<p class="field-error inline-error" role="alert">Choose your town or use your location first.</p>`;
    $("#area").focus();
    return;
  }
  readProfile();
  const body = {
    condition: state.condition,
    abstain: state.abstain,
    profile: state.profile,
    after_hours: $("#after-hours").checked,
    ...(area === "__here" && state.coords ? state.coords : { area }),
  };
  const btn = $("#run-agent");
  busy(btn, true);
  out.innerHTML = `<div class="loading"><div class="skeleton-row"><span class="sk-col"><span class="sk sk-h"></span><span class="sk sk-l"></span><span class="sk sk-l short"></span></span></div><p class="muted">Gigi is searching clinics and products…</p></div>`;
  try {
    const res = await postJSON("/api/agent", body);
    renderAgent(res);
    $("#print").hidden = false;
  } catch (e) {
    out.innerHTML = `<div class="alert alert-danger" role="alert">${icon("alert")}<div><p class="alert-h">The search didn't work.</p><p>${esc(e.message)}</p></div></div>`;
  } finally {
    busy(btn, false);
  }
}

const TOOL_LABEL = {
  plan: "Planned what you need",
  estimate_cost: "Estimated your cost",
  find_clinics: "Searched for clinics",
  recommend_products: "Matched home-care products",
  safety_review: "Checked the advice for safety",
};

function renderAgent(res) {
  const trace = `
    <details class="trace" open>
      <summary>What Gigi did <small>${res.trace.length} steps · ${esc(res.planner)} planner</small></summary>
      <ol class="trace-list">${res.trace.map((t, i) => `
        <li style="--i:${i}">
          <span class="trace-dot">${icon("check")}</span>
          <div>
            <p><span class="trace-tool">${esc(t.tool)}</span><span class="trace-ms">${t.ms} ms</span></p>
            <p class="trace-res"><span class="sr-only">${esc(TOOL_LABEL[t.tool] ?? t.tool)}: </span>${esc(t.result)}</p>
          </div>
        </li>`).join("")}
      </ol>
    </details>`;

  const clinics = res.clinics.length
    ? `<ul class="clinics">${res.clinics.map((c) => `
        <li class="clinic">
          <div>
            <p class="clinic-name">${esc(c.name)}</p>
            <p class="clinic-meta">
              <span>${icon("pin")}${esc(c.area)}</span>
              <span>${c.setting === "polyclinic" ? "Polyclinic" : "Private clinic"}</span>
              <span>${esc(c.languages.join(", "))}</span>
            </p>
            <div class="clinic-why">
              ${c.why.map((w) => `<span class="tag tag-plain ${w === "Accepts CHAS" ? "tag-chas" : ""}">${esc(w)}</span>`).join("")}
              ${res.clinic_data === "sample" ? `<span class="tag tag-plain tag-sample">Sample listing</span>` : ""}
            </div>
          </div>
          <p class="clinic-dist">${c.distance_km} km<small>from ${esc(res.near)}</small></p>
          <a class="btn btn-ghost btn-sm" href="https://www.google.com/maps/search/?api=1&query=${c.lat},${c.lng}" target="_blank" rel="noopener">Directions ${icon("external")}</a>
        </li>`).join("")}</ul>`
    : `<div class="empty-cost"><strong>No matching clinics found</strong>Try a nearby town, or untick the evening and weekend filter.</div>`;

  const products = `<ul class="products">${res.products.map((p) => `
      <li class="product">
        <p class="product-name">${esc(p.name)}</p>
        <p class="product-why">${esc(p.why)}</p>
        <p class="product-look"><b>Look for:</b> ${esc(p.look_for)}</p>
      </li>`).join("")}</ul>`;

  const advice = `
    <div class="alert ${res.urgency === "soon" ? "alert-warn" : "alert-info"} advice">
      ${icon(res.urgency === "soon" ? "alert" : "info")}
      <div><p class="alert-h">Before you go</p><ul class="ticks">${res.safety.advice.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div>
    </div>`;

  $("#agent-out").innerHTML = `
    ${trace}
    <div class="act-grid">
      <section aria-labelledby="clin-h">
        <h4 id="clin-h">Clinics near ${esc(res.near)}</h4>
        <p class="act-note">${res.clinic_data === "sample" ? `These are sample listings for this prototype. Confirm opening hours and CHAS status on the <a href="https://www.chas.sg/" target="_blank" rel="noopener">CHAS website</a> before you go.` : ""}</p>
        ${clinics}
      </section>
      <section aria-labelledby="prod-h">
        <h4 id="prod-h">Home-care products</h4>
        <p class="act-note">Product types, not brands. Most pharmacies and supermarkets stock them.</p>
        ${products}
      </section>
    </div>
    ${advice}`;
}

/* ---------------- Reliability ---------------- */
function renderReliability() {
  const m = state.meta?.metrics;
  const sum = $("#rel-summary");
  const rows = $("#rel-rows");
  if (!m) {
    sum.innerHTML = `<p>Test results aren't available on this server.</p>`;
    $(".rel-table-wrap").hidden = true;
    return;
  }
  const ci = (m.macro_f1_ci || "").replace(/[\[\]]/g, "").split(",").map((s) => s.trim());
  const caries = m.recall?.caries;
  sum.innerHTML = `
    <p>On <strong>${m.n_test}</strong> test photos, its overall balanced score (macro-F1) was <span class="hl">${m.macro_f1.toFixed(2)}</span> out of 1${ci.length === 2 ? ` (95% confidence interval ${ci[0]} to ${ci[1]})` : ""}. It's right most of the time on photos like the ones it learned from.</p>
    ${caries != null ? `<p><strong>Tooth decay is the hardest to spot.</strong> It picked up ${Math.round(caries * 100)} in 100 decay photos in testing${m.external_caries_recall != null ? `, but only about ${Math.round(m.external_caries_recall * 100)} in 100 on decay photos from other sources` : ""}. A "no decay" result doesn't rule decay out.</p>` : ""}`;
  const entries = Object.entries(m.recall || {}).filter(([, v]) => v != null).sort((a, b) => b[1] - a[1]);
  rows.innerHTML = entries.map(([k, v]) => {
    const low = v < 0.85;
    return `<tr><td>${esc(label(k))}${low ? `<span class="low-note">Weakest category</span>` : ""}</td><td>${Math.round(v * 100)} in 100</td><td><span class="rel-bar ${low ? "low" : ""}" aria-hidden="true"><i style="--p:${v.toFixed(3)}"></i></span></td></tr>`;
  }).join("");
  const low = (m.robustness || []).find((r) => /low-res/i.test(r.condition));
  if (low) $("#rel-robust").textContent = `Blurry or low-resolution photos lower accuracy (score ${m.macro_f1.toFixed(2)} → ${low.macro_f1.toFixed(2)} on small, compressed photos)`;
}

/* ---------------- Reset ---------------- */
function restart() {
  clearFile();
  $("#consent").checked = false;
  $$("#emergency-list input").forEach((i) => (i.checked = false));
  onEmergencyChange();
  Object.assign(state, { maxStep: 1, screen: null, condition: null, abstain: false, estimate: null });
  $("#cost-out").innerHTML = "";
  $("#agent-out").innerHTML = "";
  $("#print").hidden = true;
  updateAnalyse();
  goTo(1);
}

/* ---------------- Wiring ---------------- */
function wire() {
  const drop = $("#drop");
  const file = $("#file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if (f) setFile(f); });
  file.addEventListener("change", () => file.files[0] && setFile(file.files[0]));
  $("#camera").addEventListener("change", (e) => e.target.files[0] && setFile(e.target.files[0]));
  document.addEventListener("paste", (e) => {
    if (state.step !== 1) return;
    const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) setFile(f);
  });
  $("#preview-change").addEventListener("click", () => { clearFile(); drop.focus(); });
  $("#consent").addEventListener("change", updateAnalyse);
  $("#emergency-list").addEventListener("change", onEmergencyChange);
  $("#analyse").addEventListener("click", analyse);
  $("#demo-run").addEventListener("click", demoPhoto);

  document.addEventListener("click", (e) => {
    const g = e.target.closest("[data-goto]");
    if (!g) return;
    const n = Number(g.dataset.goto);
    if (n === 1 && g.closest("#s2-retake")) clearFile();
    goTo(n);
  });
  $$(".stepper button").forEach((b) => b.addEventListener("click", () => goTo(Number(b.closest("li").dataset.stepNav))));

  $("#profile").addEventListener("change", () => { loadEstimate(); });
  $("#where").addEventListener("submit", runAgent);
  $("#locate").addEventListener("click", locate);
  $("#area").addEventListener("change", () => { if ($("#area").value !== "__here") $("#locate-msg").textContent = ""; });
  $("#print").addEventListener("click", () => window.print());
  $("#restart").addEventListener("click", restart);
}

async function init() {
  wire();
  goTo(1, { focus: false });
  updateAnalyse();
  try {
    state.meta = await request("/api/meta");
  } catch (e) {
    toast(e.message);
  }
  renderEmergency();
  renderReliability();
  loadAreas();
  if (state.meta?.mode === "demo") {
    const pill = $("#mode-pill");
    pill.textContent = "Demo mode";
    pill.title = "The trained model isn't loaded on this server. Results are simulated.";
    pill.hidden = false;
    $("#demo-box").hidden = false;
    $("#demo-run").hidden = false;
  }
  const demo = new URLSearchParams(location.search).get("demo");
  if (demo && state.meta?.mode === "demo") {
    $("#scenario").value = demo;
  }
}

init();
