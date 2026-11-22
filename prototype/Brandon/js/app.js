/* SmileCheck SG prototype. `predict()` sends photos to the classifier in app/backend;
   the sample chips below still show simulated results. */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const icons = () => window.lucide && lucide.createIcons();

  const ANGLES = [
    { key: "front", label: "Front", required: true },
    { key: "upper", label: "Upper" },
    { key: "lower", label: "Lower" },
    { key: "left", label: "Left" },
    { key: "right", label: "Right" },
  ];
  const MAX_PHOTOS = ANGLES.length;
  const LOW_CONFIDENCE = 0.55;

  const state = {
    photos: {},          // angle key -> { url, file }
    targetSlot: null,    // angle key a single-slot pick goes into
    result: null,        // { key, probs, demo }
    qcKey: null,         // angle whose checklist the (i) panel shows
    retakeSlot: null,    // angle the "Try another photo" button replaces
    location: { ...DEFAULT_LOCATION, shared: false },
    filter: "best",
    activeClinic: null,
  };

  /* ---------------- Static art ---------------- */
  $("#heroArt").innerHTML = Art.heroTeeth();
  $("#dzArt").innerHTML = Art.dropArt();
  $("#emptyArt").innerHTML = Art.heroTeeth();

  /* ---------------- Angle slots ---------------- */
  function renderAngles() {
    const locked = !consented();
    $("#angles").innerHTML = ANGLES.map((a) => {
      const p = state.photos[a.key];
      return `<div class="angle ${p ? "is-filled" : ""} ${a.required ? "is-required" : ""}">
        <button type="button" class="angle-btn" data-angle="${a.key}" aria-disabled="${locked}"
            aria-label="${p ? `Replace ${a.label} photo` : `Add ${a.label} photo`}">
          <span class="angle-tile">${p ? `<img src="${p.url}" alt="">` : Art.angle(a.key)}</span>
          <span class="angle-label">${a.label}</span>
        </button>
        ${p ? `<button type="button" class="angle-remove" data-remove="${a.key}" aria-label="Remove ${a.label} photo"><i data-lucide="x"></i></button>` : ""}
        ${p ? qcBadge(a, p) : ""}
      </div>`;
    }).join("");
    icons();
    const n = Object.keys(state.photos).length;
    $("#btnAnalyse").setAttribute("aria-disabled", String(n === 0));
    $("#dzHint").textContent = n ? `${n} of ${MAX_PHOTOS} photos added` : "JPG or PNG · up to 5 photos";
    renderQc();
  }

  $("#angles").addEventListener("click", (e) => {
    const badge = e.target.closest("[data-qc]");
    if (badge) {
      state.qcKey = badge.dataset.qc;
      return openQc(true);
    }
    const rm = e.target.closest("[data-remove]");
    if (rm) {
      const k = rm.dataset.remove;
      URL.revokeObjectURL(state.photos[k].url);
      delete state.photos[k];
      if (state.qcKey === k) state.qcKey = null;
      renderAngles();
      return;
    }
    const btn = e.target.closest("[data-angle]");
    if (!btn) return;
    if (!consented()) return flashConsent();
    state.targetSlot = btn.dataset.angle;
    (isTouch() ? $("#cameraInput") : $("#fileInput")).click();
  });

  function addFiles(files) {
    const images = [...files].filter((f) => f.type.startsWith("image/"));
    if (!images.length) return;
    if (state.targetSlot) {
      setPhoto(state.targetSlot, images[0]);
      state.targetSlot = null;
    } else {
      for (const f of images) {
        const slot = ANGLES.find((a) => !state.photos[a.key]);
        if (!slot) break;
        setPhoto(slot.key, f);
      }
    }
    renderAngles();
  }

  function setPhoto(key, file) {
    if (state.photos[key]) URL.revokeObjectURL(state.photos[key].url);
    state.photos[key] = { file, url: URL.createObjectURL(file), quality: null, qcState: "checking" };
    state.qcKey = key;
    checkPhoto(key, file);
  }

  /* ---------------- Photo checklist (models/quality_check) ---------------- */
  // Same ids as the backend. The backend decides pass / warn / fail; this is only the wording.
  const QC_ITEMS = [
    { id: "size", label: "Big enough", desc: "A normal camera photo, not a tiny thumbnail" },
    { id: "light", label: "Bright enough", desc: "Your teeth are well lit" },
    { id: "exposure", label: "Not washed out", desc: "Not too bright from a lamp or sunlight" },
    { id: "glare", label: "No flash glare", desc: "No white shine hiding your teeth" },
    { id: "focus", label: "In focus", desc: "Sharp, not blurry or shaky" },
    { id: "contrast", label: "Clear detail", desc: "Close enough to make out each tooth" },
    { id: "mouth", label: "Shows teeth and gums", desc: "Lips pulled back, mouth fills the photo" },
  ];
  const QC_SHORT = {
    size: "too small", light: "too dark", exposure: "washed out", glare: "has flash glare",
    focus: "blurry", contrast: "lacks detail", mouth: "doesn't clearly show teeth",
  };
  const QC_ICON = { pass: "check", warn: "triangle-alert", fail: "x", skip: "minus" };
  const QC_SAY = { pass: "OK", warn: "needs a look", fail: "failed", skip: "not checked" };
  const angleLabel = (k) => ANGLES.find((a) => a.key === k).label;

  async function checkPhoto(key, file) {
    const form = new FormData();
    form.append("file", file);
    let q = null;
    try {
      const r = await fetch(`${API_BASE}/api/quality`, { method: "POST", body: form });
      if (r.ok) q = await r.json();
    } catch (err) { /* service offline: the check still runs when Analyse is pressed */ }
    const p = state.photos[key];
    if (!p || p.file !== file) return; // replaced or removed while checking
    p.quality = q;
    p.qcState = q ? q.status : "offline";
    renderAngles();
  }

  function qcBadge(a, p) {
    const st = p.qcState;
    if (st === "offline") return "";
    const issues = p.quality ? p.quality.checks.filter((c) => c.status === "fail" || c.status === "warn").map((c) => QC_SHORT[c.id]) : [];
    const label = st === "checking" ? `Checking ${a.label} photo`
      : st === "pass" ? `${a.label} photo passed every check` : `${a.label} photo: ${issues.join(", ")}`;
    return `<button type="button" class="qc-badge is-${st}" data-qc="${a.key}" aria-label="${label}. Open checklist" title="${label}">
      ${st === "checking" ? `<span class="qc-spin" aria-hidden="true"></span>` : `<i data-lucide="${QC_ICON[st]}"></i>`}</button>`;
  }

  function renderQc() {
    const key = state.qcKey && state.photos[state.qcKey] ? state.qcKey : ANGLES.find((a) => state.photos[a.key])?.key;
    const p = key && state.photos[key];
    const q = p && p.quality;

    if (q) {
      const bad = q.checks.some((c) => c.status === "fail");
      const meh = q.checks.some((c) => c.status === "warn");
      $("#qcHead").textContent = `${angleLabel(key)} photo`;
      $("#qcSub").textContent = bad ? "This photo will be left out. Fix the red items and take it again."
        : meh ? "Usable, but the amber items may make the result less reliable." : "All clear. This photo is good to use.";
    } else {
      $("#qcHead").textContent = "Photo checklist";
      $("#qcSub").textContent = p && p.qcState === "checking" ? `Checking your ${angleLabel(key).toLowerCase()} photo…`
        : "Every photo is checked for these before we look at your teeth.";
    }
    $("#qcList").innerHTML = QC_ITEMS.map((it) => {
      const c = q ? q.checks.find((x) => x.id === it.id) : null;
      const st = c ? c.status : "todo";
      const text = c && c.tip ? c.tip : it.desc;
      return `<li class="qc-item is-${st}">
        <span class="qc-mark" aria-hidden="true">${c ? `<i data-lucide="${QC_ICON[st]}"></i>` : ""}</span>
        <span class="qc-text"><strong>${it.label}</strong>${c ? `<span class="sr-only">: ${QC_SAY[st]}.</span>` : ""}<span class="qc-desc">${text}</span></span>
      </li>`;
    }).join("");

    // A one-line summary on the (i) chip itself, so the checklist rarely needs opening.
    const done = ANGLES.map((a) => state.photos[a.key]).filter((x) => x && x.quality);
    const failing = done.filter((x) => x.quality.status === "fail").length;
    $("#qcBtn").classList.toggle("is-fail", failing > 0);
    $("#qcBtn").classList.toggle("is-pass", done.length > 0 && !failing);
    $("#qcBtnText").textContent = !done.length ? "What we check"
      : failing ? `${failing} photo${failing > 1 ? "s" : ""} to retake` : "Photos look good";
    icons();
  }

  // Opens on hover (mouse), tap or keyboard. A tap or click pins it open until you tap elsewhere or press Esc.
  let qcPinned = false;
  let qcTimer = 0;
  function openQc(pin) {
    clearTimeout(qcTimer);
    if (pin) qcPinned = true;
    renderQc();
    $("#qcPanel").hidden = false;
    $("#qcBtn").setAttribute("aria-expanded", "true");
  }
  function closeQc(force) {
    if (qcPinned && !force) return;
    qcPinned = false;
    $("#qcPanel").hidden = true;
    $("#qcBtn").setAttribute("aria-expanded", "false");
  }
  const hoverable = () => matchMedia("(hover: hover)").matches;
  $("#qc").addEventListener("mouseenter", () => hoverable() && openQc(false));
  $("#qc").addEventListener("mouseleave", () => { if (hoverable()) qcTimer = setTimeout(() => closeQc(false), 150); });
  $("#qcBtn").addEventListener("click", () => (qcPinned ? closeQc(true) : openQc(true)));
  document.addEventListener("click", (e) => {
    if (!$("#qcPanel").hidden && !e.target.closest("#qc, [data-qc]")) closeQc(true);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#qcPanel").hidden) { closeQc(true); $("#qcBtn").focus(); }
  });

  /* ---------------- Consent (PDPA) ---------------- */
  const consented = () => $("#consent").checked;
  function syncConsent() {
    const ok = consented();
    $("#btnUpload").setAttribute("aria-disabled", String(!ok));
    $("#btnCamera").setAttribute("aria-disabled", String(!ok));
    $("#dropzone").classList.toggle("is-locked", !ok);
    $("#consentHint").hidden = ok;
    $(".consent").classList.remove("is-nudged");
    renderAngles();
  }
  $("#consent").addEventListener("change", syncConsent);

  const isTouch = () => matchMedia("(pointer: coarse)").matches;
  if (isTouch()) $("#uploadLabel").textContent = "Choose photos";

  $("#btnUpload").addEventListener("click", () => {
    if (!consented()) return flashConsent();
    state.targetSlot = null;
    $("#fileInput").click();
  });
  $("#btnCamera").addEventListener("click", () => {
    if (!consented()) return flashConsent();
    state.targetSlot = ANGLES.find((a) => !state.photos[a.key])?.key || "front";
    $("#cameraInput").click();
  });
  $("#fileInput").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  $("#cameraInput").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });

  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => {
    e.preventDefault();
    if (consented()) dz.classList.add("is-over");
  }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, () => dz.classList.remove("is-over")));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!consented()) return flashConsent();
    state.targetSlot = null;
    addFiles(e.dataTransfer.files);
  });

  function flashConsent() {
    const c = $(".consent");
    c.classList.add("is-nudged");
    c.animate([{ transform: "translateX(0)" }, { transform: "translateX(-6px)" }, { transform: "translateX(6px)" }, { transform: "translateX(0)" }], { duration: 300 });
    $("#consent").focus();
  }

  /* ---------------- Model (backend) ---------------- */
  // Served by app/backend/main.py. When the page comes from another local server, point at the backend's port.
  const API_BASE = location.port === "8000" ? "" : "http://127.0.0.1:8000";
  // shared/conditions.yaml names -> keys used by this page's content
  const MODEL_KEYS = {
    calculus: "calculus", caries: "caries", hypodontia: "hypodontia",
    mouth_ulcer: "ulcer", normal: "normal", tooth_discoloration: "discolouration",
  };

  // Sends the original files (full 0-255 pixels, no scaling). The backend checks each photo first
  // (models/quality_check), leaves out any that fail, and averages the classifier over the rest.
  // photoList items are { key, photo } in angle order.
  async function predict(photoList) {
    const form = new FormData();
    photoList.forEach((x) => form.append("files", x.photo.file));
    let out;
    try {
      const r = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: form });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      out = await r.json();
    } catch (err) {
      console.error("Prediction failed:", err);
      return { unclear: true, title: "We couldn't check this photo", why: "The screening service isn't responding. Please try again in a moment." };
    }
    // Keep each thumbnail's checklist in step with what the backend decided.
    const qc = photoList.map((x, i) => ({ ...x, q: out.quality ? out.quality[i] : null }));
    qc.forEach((x) => { if (x.q) { x.photo.quality = x.q; x.photo.qcState = x.q.status; } });
    renderAngles();
    if (out.rejected) {
      const first = qc.find((x) => x.q && !x.q.ok) || qc[0];
      return {
        unclear: true, retake: first.key, url: first.photo.url,
        title: qc.length > 1 ? "None of these photos can be read" : "We can't read this photo",
        why: first.q.checks.find((c) => c.status === "fail").tip,
      };
    }
    const used = qc.filter((x) => !x.q || x.q.ok);
    const left = qc.filter((x) => x.q && !x.q.ok);
    const weak = used.filter((x) => x.q && x.q.status === "warn");
    const notes = [];
    if (left.length) notes.push(`Left out: ${left.map((x) => `${angleLabel(x.key).toLowerCase()} photo (${firstIssue(x.q, "fail")})`).join(", ")}.`);
    if (weak.length) notes.push(`${weak.map((x) => `${angleLabel(x.key)} photo ${firstIssue(x.q, "warn")}`).join("; ")}, so this result may be less reliable.`);
    const probs = {};
    for (const [name, p] of Object.entries(out.probs)) probs[MODEL_KEYS[name]] = p;
    return { key: MODEL_KEYS[out.class_name], probs, count: used.length, photoUrl: used[0].photo.url, qcNote: notes.join(" ") };
  }

  const firstIssue = (q, st) => QC_SHORT[q.checks.find((c) => c.status === st).id];

  function makeProbs(top, topP, rand = Math.random) {
    const rest = CLASS_ORDER.filter((k) => k !== top);
    const w = rest.map(() => rand() + 0.15);
    const sum = w.reduce((a, b) => a + b, 0);
    const probs = { [top]: topP };
    rest.forEach((k, i) => (probs[k] = (w[i] / sum) * (1 - topP)));
    return probs;
  }

  function mulberry(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------------- Analyse flow ---------------- */
  $("#btnAnalyse").addEventListener("click", async () => {
    const list = ANGLES.filter((a) => state.photos[a.key]).map((a) => ({ key: a.key, photo: state.photos[a.key] }));
    if (!list.length) return;
    if ($("#btnAnalyse").getAttribute("aria-disabled") === "true") return;
    closeQc(true);
    showLoading(list[0].photo.url);
    const [res] = await Promise.all([predict(list), wait(1600)]);
    if (res.unclear) {
      state.retakeSlot = res.retake || "front";
      return showUnclear(res.url || list[0].photo.url, res.why, res.title);
    }
    showResult({ ...res, count: res.count || list.length, photoUrl: res.photoUrl || list[0].photo.url });
  });

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  function showOnly(id) {
    ["#resultEmpty", "#resultLoading", "#resultBody", "#resultUnclear"].forEach((s) => ($(s).hidden = s !== id));
    $("#resultCard").classList.toggle("is-empty", id === "#resultEmpty");
  }

  function scrollToResult() {
    const r = $("#resultCard").getBoundingClientRect();
    if (r.top < 90 || r.top > innerHeight * 0.5) $("#resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showLoading(url) {
    showOnly("#resultLoading");
    $("#scanImg").src = url;
    scrollToResult();
  }

  function showUnclear(url, why, title = "We can't read this photo") {
    showOnly("#resultUnclear");
    $("#unclearPhoto").innerHTML = `<img src="${url}" alt="The photo you added">`;
    $("#unclearTitle").textContent = title;
    $("#unclearWhy").textContent = why;
    clearResult();
  }

  $("#btnRetake").addEventListener("click", () => {
    state.targetSlot = state.retakeSlot || "front";
    (isTouch() ? $("#cameraInput") : $("#fileInput")).click();
  });

  function showResult(res) {
    state.result = res;
    const c = CONDITIONS[res.key];
    const pct = Math.round(res.probs[res.key] * 100);

    showOnly("#resultBody");

    const photo = $("#resultPhoto");
    photo.classList.toggle("is-art", !res.photoUrl);
    photo.innerHTML = res.photoUrl ? `<img src="${res.photoUrl}" alt="Your uploaded photo">` : Art.condition(res.key);
    $("#resultName").textContent = c.name;
    $("#resultAka").textContent = c.aka;
    $("#simTag").hidden = !res.demo;
    $("#photoCount").textContent = res.demo ? "this sample" : res.count > 1 ? `your ${res.count} photos` : "your photo";
    $("#lowConf").hidden = res.probs[res.key] >= LOW_CONFIDENCE;
    $("#qcNote").hidden = !res.qcNote;
    $("#qcNoteText").textContent = res.qcNote || "";

    const ring = $("#ring");
    ring.style.setProperty("--rc", c.color);
    ring.setAttribute("aria-label", `${pct} percent likelihood`);
    animateNumber($("#ringVal"), pct, (v) => ring.style.setProperty("--p", v));

    const sorted = CLASS_ORDER.slice().sort((a, b) => res.probs[b] - res.probs[a]);
    const healthy = res.key === "normal";
    $("#probs").hidden = healthy;
    $("#probsFold").hidden = !healthy;
    $("#probsFold").open = false;
    $(healthy ? "#probsFoldBody" : "#probs").innerHTML = sorted.map((k) => {
      const v = Math.round(res.probs[k] * 100);
      return `<div class="prob ${k === res.key ? "is-top" : ""}">
        <span class="cond-dot" style="--c:${CONDITIONS[k].color}" aria-hidden="true"></span>
        <span>${CONDITIONS[k].name}</span>
        <span class="prob-bar"><span style="--c:${CONDITIONS[k].color}" data-w="${v}"></span></span>
        <span class="prob-val">${v}%</span>
      </div>`;
    }).join("");
    requestAnimationFrame(() => $$(".prob-bar span").forEach((s) => (s.style.width = s.dataset.w + "%")));

    $("#consentHint").hidden = true;
    renderUnderstand(res.key);
    renderAction(res.key);
    renderCost();
    renderAsk();
    $$(".needs-result").forEach((s) => {
      $("[data-lock]", s).hidden = true;
      $("[data-content]", s).hidden = false;
    });
    initMap();
    icons();
  }

  function animateNumber(el, to, onStep) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.textContent = to + "%"; onStep(to); return;
    }
    const t0 = performance.now(), dur = 900;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3), v = Math.round(to * e);
      el.textContent = v + "%"; onStep(v);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ---------------- Understanding ---------------- */
  function renderUnderstand(key) {
    const c = CONDITIONS[key];
    $("#understandTitle").textContent = key === "normal" ? "What this means" : `About ${c.name.toLowerCase()}`;
    $("#whatArt").innerHTML = Art.condition(key);
    $("#whatText").textContent = c.what;

    $("#causesTitle").textContent = c.headings?.causes || "How it happens";
    $("#flowTitle").textContent = c.headings?.flow || "If left alone";
    $("#causeList").innerHTML = c.causes.map((x) =>
      `<li><span class="cause-icon"><i data-lucide="${x.icon}"></i></span>${x.text}</li>`).join("");

    const n = c.flow.length;
    $("#flow").innerHTML = c.flow.map((s, i) => {
      const sev = s.sev ?? (n === 1 ? 0 : Math.round((i / (n - 1)) * 3));
      const col = key === "normal" ? "var(--u1)" : URGENCY_COLORS[sev];
      return `<li><span class="flow-dot" style="--s:${col}"><i data-lucide="${s.icon}"></i></span>
        <div><p class="flow-label">${s.label}</p><p class="flow-time">${s.time}</p></div></li>`;
    }).join("");

    const u = c.urgency;
    $$("#urgencyMeter span").forEach((s, i) => (s.style.background = i < u.level ? URGENCY_COLORS[u.level - 1] : ""));
    $("#urgencyLabels").innerHTML = URGENCY_LABELS.map((l, i) => `<span class="${i === u.level - 1 ? "is-on" : ""}">${l}</span>`).join("");
    const ui = $("#urgencyIcon");
    ui.style.background = URGENCY_COLORS[u.level - 1];
    ui.innerHTML = `<i data-lucide="${u.icon}"></i>`;
    $("#urgencyTitle").textContent = u.title;
    $("#urgencyDetail").textContent = u.detail;
  }

  /* ---------------- Action: clinics + products ---------------- */
  function renderAction(key) {
    const c = CONDITIONS[key];
    $("#productsFor").textContent = `Matched to: ${c.name.toLowerCase()}`;
    $("#products").innerHTML = c.products.map((p) => {
      const item = PRODUCTS[p.id];
      const q = encodeURIComponent(item.search + " Singapore");
      return `<li><a class="product" href="https://www.google.com/search?tbm=shop&q=${q}" target="_blank" rel="noopener">
        <span class="product-art" aria-hidden="true">${Art.product(p.id)}</span>
        <div><p class="product-name">${item.name}</p><p class="product-why">${p.why}</p>
          <span class="product-link">Find in shops<i data-lucide="arrow-up-right"></i></span></div></a></li>`;
    }).join("");
    renderFilters();
    renderClinics();
  }

  function renderFilters() {
    const opts = [
      { k: "best", label: "Best for you", icon: "sparkles" },
      { k: "all", label: "All", icon: "list" },
      ...Object.entries(CLINIC_TYPES).map(([k, t]) => ({ k, label: t.label, icon: t.icon })),
    ];
    $("#filterChips").innerHTML = opts.map((o) =>
      `<button type="button" class="filter-chip" data-filter="${o.k}" aria-pressed="${state.filter === o.k}">
        <i data-lucide="${o.icon}"></i>${o.label}</button>`).join("");
  }

  $("#filterChips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-filter]");
    if (!b) return;
    state.filter = b.dataset.filter;
    renderFilters();
    renderClinics();
    icons();
  });

  // The single best match is the nearest clinic of the type that suits this result best;
  // everything else follows by distance.
  function clinicsForView() {
    const types = CONDITIONS[state.result.key].clinicTypes;
    let list = CLINICS.map((c) => ({ ...c, km: haversine(state.location, c) })).sort((a, b) => a.km - b.km);
    const bestId = list.find((c) => c.type === types[0])?.id;
    list.forEach((c) => (c.best = c.id === bestId));
    if (state.filter === "best") list = list.filter((c) => types.includes(c.type));
    else if (state.filter !== "all") list = list.filter((c) => c.type === state.filter);
    return list.sort((a, b) => (b.best - a.best) || (a.km - b.km)).slice(0, 5);
  }

  function renderClinics() {
    const list = clinicsForView();
    const far = list.length && list.every((c) => c.km > 8);
    $("#clinicFar").hidden = !far;
    $("#clinicList").innerHTML = list.map((c, i) => {
      const t = CLINIC_TYPES[c.type];
      const dest = encodeURIComponent(`${c.sample ? "CHAS dental clinic " : c.name + " "}${c.addr} Singapore`);
      return `<li class="clinic ${c.best ? "is-best" : ""} ${state.activeClinic === c.id ? "is-active" : ""}" data-clinic="${c.id}"
          tabindex="0" aria-label="${c.name}, ${c.km.toFixed(1)} kilometres. Show on map">
        <span class="clinic-num">${i + 1}</span>
        <div>
          <p class="clinic-name">${c.name}</p>
          <p class="clinic-meta"><span>${c.km.toFixed(1)} km away</span>
            ${c.best ? `<span class="clinic-tag best">Best match</span>` : ""}
            <span class="clinic-tag">${t.tag}</span></p>
          ${c.best ? `<p class="clinic-why">Closest ${c.type === "chas" ? t.label : t.label.toLowerCase()} for ${CONDITIONS[state.result.key].procedure.toLowerCase()}</p>` : ""}
          <div class="clinic-actions">
            <a class="btn ${c.best ? "btn-dark" : "btn-soft"}" href="https://www.google.com/maps/dir/?api=1&destination=${dest}" target="_blank" rel="noopener"><i data-lucide="navigation"></i>Directions</a>
            ${c.type === "polyclinic" ? `<a class="clinic-link" href="https://www.healthhub.sg/" target="_blank" rel="noopener">Book via HealthHub<i data-lucide="arrow-up-right"></i></a>` : ""}
            ${c.sample ? `<a class="clinic-link" href="https://www.chas.sg/clinic-locator" target="_blank" rel="noopener">Find on CHAS<i data-lucide="arrow-up-right"></i></a>` : ""}
          </div>
        </div>
      </li>`;
    }).join("");
    drawMarkers(list);
    icons();
  }

  $("#clinicList").addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    const li = e.target.closest("[data-clinic]");
    if (li) focusClinic(li.dataset.clinic);
  });
  $("#clinicList").addEventListener("keydown", (e) => {
    const li = e.target.closest("[data-clinic]");
    if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); focusClinic(li.dataset.clinic); }
  });

  function focusClinic(id) {
    state.activeClinic = id;
    $$(".clinic").forEach((el) => el.classList.toggle("is-active", el.dataset.clinic === id));
    const m = markers[id];
    if (m && map) {
      map.flyTo(m.getLatLng(), 14, { duration: 0.6 });
      m.openPopup();
    }
    Object.entries(markers).forEach(([k, mk]) => mk.getElement()?.querySelector(".pin")?.classList.toggle("is-active", k === id));
  }

  function haversine(a, b) {
    const R = 6371, r = Math.PI / 180;
    const dLat = (b.lat - a.lat) * r, dLng = (b.lng - a.lng) * r;
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  }

  /* ---------------- Cost estimate (app/backend -> prototype/Brandon/backend/costs.py) ---------------- */
  // Page keys -> condition names in fees.json
  const COST_KEYS = {
    calculus: "calculus", caries: "caries", hypodontia: "hypodontia",
    ulcer: "mouth_ulcer", normal: "normal", discolouration: "tooth_discoloration",
  };
  let costSeq = 0;

  const money = (v, up) => "S$" + (up ? Math.ceil(v) : Math.floor(v)).toLocaleString("en-SG");
  const range = (lo, hi) => (Math.floor(lo) === Math.ceil(hi) ? money(lo) : `${money(lo)} – ${money(hi, true)}`);
  // Exact amounts (subsidies) keep their cents: S$20.50, S$30.
  const exact = (v) => "S$" + (Number.isInteger(v) ? v : v.toFixed(2)).toLocaleString("en-SG");
  const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

  function costProfile() {
    const f = $("#costForm");
    return { residency: f.residency.value, tier: f.tier.value, age: f.age.value };
  }

  // Answers only live in this request; they are never stored.
  async function renderCost() {
    if (!state.result) return;
    const seq = ++costSeq;
    const out = $("#costOut");
    out.classList.add("is-busy");
    let est;
    try {
      const r = await fetch(`${API_BASE}/api/estimate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition: COST_KEYS[state.result.key], ...costProfile() }),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      est = await r.json();
    } catch (err) {
      console.error("Cost estimate failed:", err);
      if (seq !== costSeq) return;
      out.classList.remove("is-busy");
      out.innerHTML = `<div class="cost-error"><i data-lucide="cloud-off"></i><p>We couldn't load the cost estimate. Please try again in a moment.</p></div>`;
      icons();
      return;
    }
    if (seq !== costSeq) return; // a newer answer is already on its way
    out.classList.remove("is-busy");
    out.innerHTML = costHtml(est);
    icons();
  }

  function costLines(lines) {
    return `<ul class="cost-lines">${lines.map((l) => {
      const bits = [`Fee ${range(l.fee_min, l.fee_max)} before GST`];
      if (l.subsidy) bits.push(`CHAS −${exact(l.subsidy)}`);
      if (l.medisave) bits.push(`MediSave up to −${exact(l.medisave)}`);
      return `<li class="cost-line">
        <div><p class="cost-line-name">${esc(l.name)}</p>
          <p class="cost-line-plain">${esc(l.plain)}</p>
          <p class="cost-line-math">${bits.join(" · ")}</p>
          ${l.claim_limit ? `<p class="cost-line-limit"><i data-lucide="calendar"></i>CHAS: ${esc(l.claim_limit)}</p>` : ""}
          ${l.range_note ? `<p class="cost-line-limit"><i data-lucide="info"></i>${esc(l.range_note)}</p>` : ""}</div>
        <p class="cost-line-pay">${range(l.pay_min, l.pay_max)}</p>
      </li>`;
    }).join("")}</ul>`;
  }

  function costHtml(est) {
    const name = CONDITIONS[state.result.key].name.toLowerCase();
    const t = est.likely.total;
    const none = !est.likely.lines.length;
    const head = none
      ? `<p class="cost-eyebrow">For ${esc(name)}</p>
         <p class="cost-total">Usually ${money(0)}</p>
         <p class="cost-sub">There's often nothing to pay. See below if it doesn't heal.</p>`
      : `<p class="cost-eyebrow">Likely visit for ${esc(name)}</p>
         <p class="cost-total">${range(t.pay_min, t.pay_max)}</p>
         <p class="cost-sub">What you'd pay${t.subsidy ? `, after up to ${exact(t.subsidy)} off with CHAS` : ""}.
           Fees before any help: ${range(t.fee_min, t.fee_max)} + GST.</p>`;

    const later = est.later.map((s) => `
      <details class="cost-later">
        <summary><span>${esc(s.label)}</span><span class="cost-later-pay">${range(s.total.pay_min, s.total.pay_max)}</span><i data-lucide="chevron-down"></i></summary>
        ${costLines(s.lines)}
      </details>`).join("");

    const sources = Object.values(est.sources).map((s) =>
      `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}<i data-lucide="arrow-up-right"></i></a><span>checked ${esc(s.checked)}</span></li>`).join("");

    return `
      <span class="estimate-pill"><i data-lucide="badge-info"></i>Estimate only, not a quote</span>
      ${head}
      ${none ? "" : costLines(est.likely.lines)}
      ${later ? `<p class="cost-later-title">Other treatments a dentist might suggest</p>${later}` : ""}
      <ul class="cost-notes">${est.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
      <details class="cost-sources"><summary><i data-lucide="book-open"></i>Where these figures come from</summary><ul>${sources}</ul></details>`;
  }

  $("#costForm").addEventListener("change", renderCost);

  /* ---------------- Ask about your result (app/backend -> genai/rag/answer.py) ---------------- */
  let asking = false;

  async function renderAsk() {
    $("#askThread").innerHTML = "";
    $("#askInput").value = "";
    let qs = [];
    try {
      const r = await fetch(`${API_BASE}/api/ask/suggestions?condition=${COST_KEYS[state.result.key]}`);
      qs = (await r.json()).questions || [];
    } catch (err) {
      console.error("Suggestions failed:", err);
    }
    $("#askChips").innerHTML = qs.map((q) => `<button type="button" class="ask-chip" data-q="${esc(q)}">${esc(q)}</button>`).join("");
  }

  $("#askChips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-q]");
    if (b) askQuestion(b.dataset.q);
  });
  $("#askForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#askInput").value.trim();
    if (q) askQuestion(q);
  });

  async function askQuestion(q) {
    if (asking || !state.result) return;
    asking = true;
    $("#askSend").setAttribute("aria-disabled", "true");
    $("#askInput").value = "";
    const thread = $("#askThread");
    thread.insertAdjacentHTML("beforeend", `<p class="ask-q">${esc(q)}</p><div class="ask-a is-loading"><p>Looking that up…</p></div>`);
    const bubble = thread.lastElementChild;
    bubble.scrollIntoView({ behavior: "smooth", block: "nearest" });
    try {
      const r = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, condition: COST_KEYS[state.result.key], ...costProfile() }),
      });
      if (!r.ok) throw new Error(r.statusText);
      bubble.outerHTML = answerHtml(await r.json());
    } catch (err) {
      console.error("Ask failed:", err);
      bubble.outerHTML = `<div class="ask-a is-muted"><p>Sorry, we couldn't answer that just now. Please try again in a moment.</p></div>`;
    }
    icons();
    asking = false;
    $("#askSend").removeAttribute("aria-disabled");
  }

  function answerHtml(a) {
    const id = "a" + Math.random().toString(36).slice(2, 8);
    // [n] markers become small links to the matching source below the answer.
    const body = esc(a.answer).replace(/\s?\[(\d+)\]/g, (_, n) => `<sup><a href="#${id}-${n}" aria-label="Source ${n}">${n}</a></sup>`);
    const cls = a.route === "emergency" ? "is-emergency" : a.route === "answer" ? "" : "is-muted";
    const src = a.citations.length ? `<ol class="ask-src">${a.citations.map((c) => `
      <li id="${id}-${c.n}"><b>${c.n}</b><span>${esc(c.title)}${c.section ? ` · ${esc(c.section)}` : ""}<br>
        ${c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.source)}</a>` : esc(c.source)}</span></li>`).join("")}</ol>` : "";
    const links = [];
    if (a.cost) links.push(`<a href="#cost"><i data-lucide="wallet"></i>${a.cost.has_items ? `Your estimate: ${range(a.cost.pay_min, a.cost.pay_max)}` : "See your cost estimate"}</a>`);
    if (a.show_clinics) links.push(`<a href="#act"><i data-lucide="map-pin"></i>Clinics near you</a>`);
    const tag = a.route !== "answer" ? "" : a.mode === "llm"
      ? `<span class="ask-tag"><i data-lucide="sparkles"></i>Written by AI from the sources below. Check with a dentist.</span>`
      : `<span class="ask-tag"><i data-lucide="quote"></i>Quoted from the sources below.</span>`;
    return `<div class="ask-a ${cls}">
      <p>${body}</p>
      ${a.route === "emergency" ? `<a class="ask-call" href="tel:995"><i data-lucide="phone"></i>Call 995</a>` : ""}
      ${links.length ? `<div class="ask-links">${links.join("")}</div>` : ""}
      ${a.notes.map((n) => `<p class="ask-note">${esc(n)}</p>`).join("")}
      ${tag}${src}
    </div>`;
  }

  /* ---------------- Map ---------------- */
  let map = null, youMarker = null, markers = {};

  function initMap() {
    if (!window.L) return;
    if (!map) {
      map = L.map("map", { scrollWheelZoom: false, zoomControl: true }).setView([state.location.lat, state.location.lng], 12);
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
    }
    placeYou();
    renderClinics();
    setTimeout(() => { map.invalidateSize(); fitAll(); }, 60);
  }

  function placeYou() {
    if (!map) return;
    const icon = L.divIcon({ className: "", html: '<div class="you-dot"></div>', iconSize: [22, 22], iconAnchor: [11, 11] });
    if (youMarker) youMarker.setLatLng([state.location.lat, state.location.lng]);
    else youMarker = L.marker([state.location.lat, state.location.lng], { icon, title: "You", zIndexOffset: 1000 }).addTo(map).bindPopup("You are here");
  }

  function drawMarkers(list) {
    if (!map) return;
    Object.values(markers).forEach((m) => m.remove());
    markers = {};
    list.forEach((c, i) => {
      const icon = L.divIcon({
        className: "",
        html: `<div class="pin ${c.best ? "is-best" : ""} ${state.activeClinic === c.id ? "is-active" : ""}"><b>${i + 1}</b></div>`,
        iconSize: [40, 40], iconAnchor: [6, 40], popupAnchor: [14, -40],
      });
      markers[c.id] = L.marker([c.lat, c.lng], { icon, title: c.name }).addTo(map)
        .bindPopup(`<strong>${c.name}</strong><br>${c.addr}<br>${c.km.toFixed(1)} km away`)
        .on("click", () => focusClinic(c.id));
    });
    fitAll();
  }

  function fitAll() {
    if (!map) return;
    const pts = Object.values(markers).map((m) => m.getLatLng());
    if (youMarker) pts.push(youMarker.getLatLng());
    if (pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [40, 40], maxZoom: 14 });
  }

  const areaSelect = $("#areaSelect");
  areaSelect.innerHTML = AREAS.map((a, i) => `<option value="${i}" ${a.label === DEFAULT_LOCATION.label ? "selected" : ""}>${a.label}</option>`).join("");
  areaSelect.addEventListener("change", () => {
    const a = AREAS[areaSelect.value];
    state.location = { lat: a.lat, lng: a.lng, label: a.label, shared: false };
    $("#mapNote").innerHTML = `<i data-lucide="info"></i>Distances from ${a.label}. Your area is never stored or sent.`;
    placeYou(); renderClinics(); icons();
  });

  $("#btnLocate").addEventListener("click", () => {
    const note = $("#mapNote");
    if (!navigator.geolocation) return;
    note.innerHTML = '<i data-lucide="loader"></i>Finding you…';
    icons();
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        state.location = { lat: pos.coords.latitude, lng: pos.coords.longitude, label: "you", shared: true };
        note.innerHTML = '<i data-lucide="locate-fixed"></i>Distances from your location. It stays on this device and is never stored.';
        placeYou(); renderClinics(); icons();
      },
      () => {
        note.innerHTML = `<i data-lucide="info"></i>Couldn't get your location. Showing distances from ${DEFAULT_LOCATION.label}.`;
        icons();
      },
      { timeout: 8000, maximumAge: 60000 },
    );
  });

  /* ---------------- Demo samples ---------------- */
  $("#demoChips").innerHTML = CLASS_ORDER.map((k) =>
    `<button type="button" class="demo-chip" data-demo="${k}"><span class="cond-dot" style="--c:${CONDITIONS[k].color}"></span>${CONDITIONS[k].name}</button>`).join("");
  $("#demoChips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-demo]");
    if (b) runDemo(b.dataset.demo, true);
  });

  async function runDemo(key, animate) {
    if (animate) {
      showLoading(svgUrl(Art.condition(key)));
      await wait(1100);
    }
    const seedRand = mulberry(key.length * 977 + key.charCodeAt(0));
    showResult({ key, probs: makeProbs(key, 0.72 + seedRand() * 0.18, seedRand), count: 1, demo: true, photoUrl: null });
  }

  const svgUrl = (svg) => "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);

  /* ---------------- Reset ---------------- */
  function clearResult() {
    state.result = null;
    $$(".needs-result").forEach((s) => {
      $("[data-lock]", s).hidden = false;
      $("[data-content]", s).hidden = true;
    });
    $("#costOut").innerHTML = "";
    $("#askThread").innerHTML = "";
  }

  $("#btnReset").addEventListener("click", () => {
    Object.values(state.photos).forEach((p) => URL.revokeObjectURL(p.url));
    state.photos = {};
    clearResult();
    showOnly("#resultEmpty");
    renderAngles();
    $("#screen").scrollIntoView({ behavior: "smooth" });
  });

  /* ---------------- Nav highlight + reveal ---------------- */
  const navObs = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) $$(".nav-chips .chip").forEach((c) => c.classList.toggle("is-active", c.getAttribute("href") === "#" + en.target.id));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  $$("main section[id]").forEach((s) => navObs.observe(s));

  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("is-in"); revealObs.unobserve(en.target); } });
  }, { threshold: 0.12 });
  $$(".section-head, .card, .safety-item, .locked").forEach((el, i) => {
    el.classList.add("reveal");
    el.style.transitionDelay = `${(i % 3) * 70}ms`;
    revealObs.observe(el);
  });

  /* ---------------- Boot ---------------- */
  syncConsent();
  icons();
  const demo = new URLSearchParams(location.search).get("demo");
  if (demo && CONDITIONS[demo]) runDemo(demo, false);
})();
