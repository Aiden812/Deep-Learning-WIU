/* SmileCheck SG prototype: front-end only. The model call is simulated in `predict()`;
   swap it for the real CNN endpoint later without touching the rendering code. */
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
      </div>`;
    }).join("");
    icons();
    const n = Object.keys(state.photos).length;
    $("#btnAnalyse").setAttribute("aria-disabled", String(n === 0));
    $("#dzHint").textContent = n ? `${n} of ${MAX_PHOTOS} photos added` : "JPG or PNG · up to 5 photos";
  }

  $("#angles").addEventListener("click", (e) => {
    const rm = e.target.closest("[data-remove]");
    if (rm) {
      const k = rm.dataset.remove;
      URL.revokeObjectURL(state.photos[k].url);
      delete state.photos[k];
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
    state.photos[key] = { file, url: URL.createObjectURL(file) };
  }

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

  /* ---------------- Simulated model ---------------- */
  // Reads a tiny thumbnail of each photo and turns its colour statistics into a stable
  // pseudo-probability over the six dataset classes. Placeholder for the trained CNN.
  async function predict(photoList) {
    let seed = 0;
    for (const [n, p] of photoList.entries()) {
      const img = await loadImage(p.url);
      const S = 32;
      const c = document.createElement("canvas");
      c.width = c.height = S;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, S, S);
      const d = ctx.getImageData(0, 0, S, S).data;
      for (let i = 0; i < d.length; i += 4) seed = (seed * 31 + d[i] + d[i + 1] * 3 + d[i + 2] * 7) >>> 0;
      if (n === 0) {
        const why = photoQuality(d, S);
        if (why) return { unclear: true, why };
      }
    }
    const rand = mulberry(seed);
    const top = CLASS_ORDER[Math.floor(rand() * CLASS_ORDER.length)];
    return { key: top, probs: makeProbs(top, 0.6 + rand() * 0.32, rand) };
  }

  // Rejects photos the model should not guess on: too dark, washed out, flat or blurry.
  // Returns a plain-language reason, or null when the photo is usable.
  function photoQuality(d, S) {
    const lum = [];
    for (let i = 0; i < d.length; i += 4) lum.push(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    const mean = lum.reduce((a, b) => a + b, 0) / lum.length;
    const sd = Math.sqrt(lum.reduce((a, b) => a + (b - mean) ** 2, 0) / lum.length);
    let edges = 0;
    for (let y = 0; y < S; y++) for (let x = 1; x < S; x++) edges += Math.abs(lum[y * S + x] - lum[y * S + x - 1]);
    edges /= S * (S - 1);
    if (mean < 45) return "It's too dark. Try near a window or turn on a light.";
    if (mean > 238) return "It's too bright or washed out. Avoid pointing at a lamp.";
    if (sd < 14 || edges < 3) return "We can't see any teeth. Pull your lips back and get closer.";
    return null;
  }

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

  const loadImage = (src) => new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = rej;
    i.src = src;
  });

  /* ---------------- Analyse flow ---------------- */
  $("#btnAnalyse").addEventListener("click", async () => {
    const list = ANGLES.map((a) => state.photos[a.key]).filter(Boolean);
    if (!list.length) return;
    if ($("#btnAnalyse").getAttribute("aria-disabled") === "true") return;
    showLoading(list[0].url);
    const [res] = await Promise.all([predict(list), wait(1600)]);
    if (res.unclear) return showUnclear(list[0].url, res.why);
    showResult({ ...res, count: list.length, photoUrl: list[0].url });
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

  function showUnclear(url, why) {
    showOnly("#resultUnclear");
    $("#unclearPhoto").innerHTML = `<img src="${url}" alt="The photo you added">`;
    $("#unclearWhy").textContent = why;
    clearResult();
  }

  $("#btnRetake").addEventListener("click", () => {
    state.targetSlot = "front";
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
    $("#photoCount").textContent = res.demo ? "this sample" : res.count > 1 ? `your ${res.count} photos` : "your photo";
    $("#lowConf").hidden = res.probs[res.key] >= LOW_CONFIDENCE;

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
    $("#costProcedure").textContent = c.procedure;
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
    $("#costProcedure").textContent = "Treatment";
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
