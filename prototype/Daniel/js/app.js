/* SmileCheck SG prototype. `predict()` sends photos and videos to prototype/Daniel/server.py
   (Daniel's mouth gate + issue gate + YOLO detector, every video frame included); the sample
   chips below still show simulated results. */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const icons = () => window.lucide && lucide.createIcons();

  const LOW_CONFIDENCE = 0.55;
  const VIDEO_EXT = /\.(mp4|mov|m4v|webm|avi|mkv|3gp)$/i;

  const state = {
    media: [],           // { id, file, kind: "image" | "video", url }
    result: null,        // { key, probs, demo }
    location: { ...DEFAULT_LOCATION, shared: false },
    filter: "best",
    activeClinic: null,
  };
  let nextId = 1;

  /* ---------------- Static art ---------------- */
  $("#heroArt").innerHTML = Art.heroTeeth();
  $("#emptyArt").innerHTML = Art.heroTeeth();

  /* ---------------- Photos and videos ---------------- */
  const isTouch = () => matchMedia("(pointer: coarse)").matches;
  const pasteKey = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘V" : "Ctrl+V";

  function kindOf(f) {
    if (f.type.startsWith("image/")) return "image";
    if (f.type.startsWith("video/") || VIDEO_EXT.test(f.name)) return "video";
    return null;
  }

  // "Photo", "Video" when there is one of a kind; "Photo 1", "Photo 2"... otherwise.
  function labelled(list = state.media) {
    const total = { image: 0, video: 0 }, seen = { image: 0, video: 0 };
    list.forEach((m) => total[m.kind]++);
    return list.map((m) => {
      const n = ++seen[m.kind], word = m.kind === "image" ? "Photo" : "Video";
      return { ...m, label: total[m.kind] > 1 ? `${word} ${n}` : word };
    });
  }

  // "2 photos and 1 video"
  function describe(list) {
    const n = { image: 0, video: 0 };
    list.forEach((m) => n[m.kind]++);
    return [["image", "photo"], ["video", "video"]].filter(([k]) => n[k])
      .map(([k, w]) => `${n[k]} ${w}${n[k] > 1 ? "s" : ""}`).join(" and ");
  }

  const mediaHtml = (m, alt = "") => m.kind === "video"
    ? `<video src="${m.url}#t=0.1" muted playsinline preload="metadata" aria-label="${alt}"></video>`
    : `<img src="${m.url}" alt="${alt}">`;

  function addFiles(files) {
    if (!consented()) return flashConsent();
    const add = [...files].filter(kindOf);
    if (!add.length) return;
    add.forEach((file) => state.media.push({ id: nextId++, file, kind: kindOf(file), url: URL.createObjectURL(file) }));
    renderMedia();
  }

  function removeMedia(id) {
    const i = state.media.findIndex((m) => m.id === id);
    if (i < 0) return;
    URL.revokeObjectURL(state.media[i].url);
    state.media.splice(i, 1);
    renderMedia();
  }

  function clearMedia() {
    state.media.forEach((m) => URL.revokeObjectURL(m.url));
    state.media = [];
    renderMedia();
  }

  function renderMedia() {
    const list = labelled();
    $("#media").hidden = !list.length;
    $("#media").innerHTML = list.map((m) => `<li class="media-item">
        ${mediaHtml(m, m.label)}
        ${m.kind === "video" ? `<span class="media-badge" aria-hidden="true"><i data-lucide="video"></i></span>` : ""}
        <button type="button" class="media-remove" data-remove="${m.id}" aria-label="Remove ${m.label}"><i data-lucide="x"></i></button>
      </li>`).join("") +
      `<li><button type="button" class="media-add" data-add aria-label="Add more"><i data-lucide="plus"></i></button></li>`;
    $("#btnAnalyse").setAttribute("aria-disabled", String(!list.length));
    syncHint();
    icons();
  }

  function syncHint() {
    const n = state.media.length;
    $("#dzHint").textContent = !consented() ? "Turn on the switch above first"
      : n ? `${describe(state.media)} added`
      : isTouch() ? "Photos or videos of your teeth" : `Photos or videos · ${pasteKey} to paste`;
  }

  $("#media").addEventListener("click", (e) => {
    const rm = e.target.closest("[data-remove]");
    if (rm) return removeMedia(Number(rm.dataset.remove));
    if (e.target.closest("[data-add]")) $("#fileInput").click();
  });

  /* ---------------- Consent (PDPA) ---------------- */
  const consented = () => $("#consent").checked;
  function syncConsent() {
    const ok = consented();
    $("#btnUpload").setAttribute("aria-disabled", String(!ok));
    $("#btnCamera").setAttribute("aria-disabled", String(!ok));
    $("#dropzone").classList.toggle("is-locked", !ok);
    $(".consent").classList.remove("is-nudged");
    syncHint();
  }
  $("#consent").addEventListener("change", syncConsent);

  $("#btnUpload").addEventListener("click", () => consented() ? $("#fileInput").click() : flashConsent());
  $("#btnCamera").addEventListener("click", () => consented() ? $("#cameraInput").click() : flashConsent());
  $("#fileInput").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  $("#cameraInput").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });

  const dz = $("#uploadCard");
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => {
    e.preventDefault();
    if (consented()) $("#dropzone").classList.add("is-over");
  }));
  dz.addEventListener("dragleave", (e) => { if (!dz.contains(e.relatedTarget)) $("#dropzone").classList.remove("is-over"); });
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    $("#dropzone").classList.remove("is-over");
    addFiles(e.dataTransfer.files);
  });

  // Paste anywhere on the page (screenshots, copied photos, copied files).
  document.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (!files.some(kindOf)) return;
    e.preventDefault();
    addFiles(files);
    if (consented()) {
      $("#dropzone").animate([{ background: "var(--peach)" }, { background: "var(--surface)" }], { duration: 600 });
    }
  });

  function flashConsent() {
    const c = $(".consent");
    c.classList.add("is-nudged");
    c.scrollIntoView({ behavior: "smooth", block: "center" });
    c.animate([{ transform: "translateX(0)" }, { transform: "translateX(-6px)" }, { transform: "translateX(6px)" }, { transform: "translateX(0)" }], { duration: 300 });
    $("#consent").focus({ preventScroll: true });
  }

  /* ---------------- Model (backend) ---------------- */
  // Served by prototype/Daniel/server.py, which runs Daniel's pipeline from models/classifier/demo/Daniel:
  // mouth gate -> issue gate (p(issue) vs its tuned threshold) -> YOLO detector on flagged photos.
  // When the page comes from another local server, point at the backend's port.
  const API_PORT = "8713";
  const API_BASE = location.port === API_PORT ? "" : `http://127.0.0.1:${API_PORT}`;
  // ?classifier=off skips the issue classifier (YOLO alone decides); ?classifier=on forces it on.
  // Without the parameter the server's PROTO_CLASSIFIER setting applies.
  const CLASSIFIER_PARAM = new URLSearchParams(location.search).get("classifier");
  // Six-class checkpoint names (if the server runs with DEMO_WEIGHTS=...best.pt) -> this page's keys.
  const MODEL_KEYS = {
    calculus: "calculus", caries: "caries", hypodontia: "hypodontia",
    mouth_ulcer: "ulcer", normal: "normal", tooth_discoloration: "discolouration",
  };
  // YOLO detector classes -> this page's keys.
  const DET_KEYS = {
    Calculus: "calculus", Caries: "caries", Hypodontia: "hypodontia",
    Mouth_Ulcer: "ulcer", Tooth_Discoloration: "discolouration",
  };
  const DET_ORDER = CLASS_ORDER.filter((k) => k !== "normal");

  // Mouth-gate outcomes, in plain words: [title when nothing could be checked, reason for one skipped item].
  const GATE_REASONS = {
    no_mouth: ["We can't see a mouth", "it doesn't show a mouth or teeth."],
    mouth_not_found: ["We can't find your mouth", "we couldn't find the mouth. Face the camera with your mouth open."],
    mouth_too_small: ["Your mouth is too small in the frame", "the mouth is too small. Move the camera closer."],
    mouth_closed: ["Your mouth looks closed", "the mouth looks closed. Open wide or smile so your teeth show."],
    unreadable: ["We couldn't open this photo", "we couldn't open this file. Try a JPG or PNG."],
    unreadable_video: ["We couldn't open this video", "we couldn't read this video. Try an MP4 or MOV."],
  };
  const RETAKE_TIP = "Take a close-up of your teeth with your lips pulled back, or a front-facing shot with your mouth open.";
  const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
  const mostCommon = (counts) => Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];

  // Drops photos that are clearly unusable here first, then streams the rest to the server,
  // which runs every photo and every video frame through the pipeline and combines them.
  async function predict(list, onProgress) {
    const send = [], skipped = [];
    for (const m of list) {
      const why = m.kind === "image" ? await quickCheck(m.url) : null;
      if (why) skipped.push({ ...m, why });
      else send.push(m);
    }
    if (!send.length) return { unclear: true, media: skipped[0], why: skipped[0].why };

    const form = new FormData();
    send.forEach((m) => form.append("files", m.file, m.file.name || (m.kind === "video" ? "video.mp4" : "photo.png")));
    if (CLASSIFIER_PARAM) form.append("classifier", CLASSIFIER_PARAM);
    let out;
    try {
      out = await streamPredict(form, onProgress);
    } catch (err) {
      console.error("Prediction failed:", err);
      return { unclear: true, media: send[0], title: "We couldn't check this", why: "The screening service isn't responding. Please try again in a moment." };
    }
    return fromModel(out, send, skipped);
  }

  // /api/predict answers with NDJSON: a start line, one progress line per photo or frame, then the result.
  async function streamPredict(form, onProgress) {
    const r = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: form });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "", result = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.type === "start" || msg.type === "progress") onProgress?.(msg.done || 0, msg.total);
        else if (msg.type === "result") result = msg;
        else if (msg.type === "error") throw new Error(msg.detail);
      }
    }
    if (!result) throw new Error("The server closed the connection without a result.");
    return result;
  }

  // Turns the server's combined result into what showResult() draws.
  // `send` is what went to the server (same order as out.sources); `skipped` failed the quick check.
  async function fromModel(out, send, skipped) {
    const reasonOf = (counts) => (GATE_REASONS[mostCommon(counts || {})] || [])[1] || "it couldn't be checked.";

    if (out.rejected) {
      const counts = {};
      out.sources.forEach((s) => Object.entries(s.reasons).forEach(([k, v]) => (counts[k] = (counts[k] || 0) + v)));
      const [title, reason] = GATE_REASONS[mostCommon(counts)] || ["We can't read this", "it couldn't be checked."];
      const why = send.length === 1 && send[0].kind === "image" ? `${cap(reason)} ${RETAKE_TIP}`
        : `None of ${send.length > 1 ? "them" : "the frames"} could be checked. Mostly, ${reason} ${RETAKE_TIP}`;
      return { unclear: true, media: send[0], title, why };
    }

    // Notes on photos skipped here or by the mouth gate, and a line per video.
    const notes = skipped.map((m) => ({ icon: "image-off", html: `<b>${m.label}</b> skipped. ${m.why}` }));
    out.sources.forEach((s) => {
      const m = send[s.index];
      if (!s.accepted) {
        notes.push({ icon: s.kind === "video" ? "video-off" : "image-off", html: `<b>${m.label}</b> skipped: ${reasonOf(s.reasons)}` });
      } else if (s.kind === "video") {
        const flagged = out.task === "binary" && s.flagged_frames ? `, ${s.flagged_frames} worth a check` : "";
        notes.push({ icon: "film", html: `<b>${m.label}</b>: ${s.accepted} of ${s.frames} frames checked${flagged}.` });
      }
    });

    const used = out.sources.filter((s) => s.accepted).map((s) => send[s.index]);
    const nImg = used.filter((m) => m.kind === "image").length, nVid = used.length - nImg;
    const res = {
      notes,
      plural: used.length > 1,
      what: used.length === 1 ? `your ${nImg ? "photo" : "video"}`
        : !nVid ? `your ${nImg} photos` : !nImg ? `your ${nVid} videos` : `your photos and video${nVid > 1 ? "s" : ""}`,
    };

    const findings = (out.findings || []).filter((f) => DET_KEYS[f.class]);
    const top = findings[0];
    const shown = out.photos.find((p) => p.index === out.show_photo);
    const shownSrc = send[shown.source];
    // A healthy verdict draws no boxes (without the classifier, YOLO's weak boxes would otherwise show).
    res.view = out.task === "binary" && !out.flagged ? null : await detectionView(shown);
    res.photoUrl = shown.image || (shownSrc.kind === "image" ? shownSrc.url : null);

    if (out.task === "binary") {
      // With the classifier off, p is the strongest YOLO box confidence, not a calibrated p(issue).
      const p = out.p_issue, detOnly = out.classifier === false;
      if (!out.flagged) {
        res.key = "normal";
        res.kicker = "Our screening says";
        res.ring = { value: 1 - p, title: "Looks healthy", text: detOnly
          ? (what) => `Nothing in ${what} was outlined clearly enough to flag. A dentist must confirm.`
          : (what, s) => `How likely ${what} show${s} none of the five conditions. A dentist must confirm.` };
        res.bars = [{ key: "normal", value: 1 - p }, { key: "check", value: p }];
        if (p >= 0.5) res.lowConf = "It's a close call. Try a brighter, closer shot, or ask a dentist at your next check-up.";
      } else {
        res.key = top ? DET_KEYS[top.class] : "check";
        res.kicker = top?.confident ? "Looks most like" : top ? "Might be" : "Our screening says";
        res.ring = { value: p, title: "Needs a check", text: detOnly
          ? (what) => `How strongly the clearest finding in ${what} was outlined.`
          : (what, s) => `How likely ${what} show${s} something a dentist should look at.` };
        if (out.detector_ran) {
          // Strongest outline per condition, pooled over the photos and frames.
          const conf = Object.fromEntries(findings.map((f) => [DET_KEYS[f.class], f.conf]));
          res.bars = DET_ORDER.map((k) => ({ key: k, value: conf[k] || 0 }));
          res.barsTitle = "How strongly each condition was outlined";
        }
        if (!top) res.lowConf = "We couldn't tell which condition it is. A dentist should take a look.";
        else if (!top.confident) res.lowConf = "We're not sure what this is. The dashed outlines are only best guesses, and it may be nothing. A check-up will tell.";
      }
    } else {
      // Six-class checkpoint: averaged probabilities, abstain below its threshold.
      res.key = MODEL_KEYS[out.class_name] || "check";
      res.bars = Object.entries(out.probs).map(([n, v]) => ({ key: MODEL_KEYS[n], value: v })).filter((b) => b.key);
      res.ring = { value: out.confidence };
      if (out.abstain) res.lowConf = "The model isn't sure. Try a brighter, closer shot.";
    }
    return res;
  }

  // The detector's image of the shown photo with its boxes (confident ones, or the best guesses
  // when nothing is confident), as an SVG so the boxes stay aligned at any size.
  async function detectionView(photo) {
    const det = photo?.detection;
    if (!det || !det.boxes.length) return null;
    const boxes = det.n_confident ? det.boxes.filter((b) => b.confident) : det.boxes;
    const img = await loadImage(det.image);
    return { src: det.image, w: img.naturalWidth, h: img.naturalHeight, boxes };
  }

  function detectionSvg(v) {
    const fs = Math.round(Math.max(v.w, v.h) * 0.032), pad = fs * 0.4, th = fs * 1.5;
    const parts = [...v.boxes].reverse().map((b) => {  // strongest drawn last, on top
      const key = DET_KEYS[b.class];
      const col = key ? CONDITIONS[key].color : "#475569";
      const [x1, y1, x2, y2] = [b.box[0] * v.w, b.box[1] * v.h, b.box[2] * v.w, b.box[3] * v.h];
      const text = `${key ? CONDITIONS[key].name : b.display_name} ${Math.round(b.conf * 100)}%`;
      const tw = text.length * fs * 0.55 + pad * 2;
      const tx = Math.max(0, Math.min(x1, v.w - tw));
      const ty = y1 >= th ? y1 - th : y1;
      return `<g style="--c:${col}">
        <rect class="det-box ${b.confident ? "" : "is-guess"}" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" rx="${fs * 0.3}"/>
        <rect class="det-tag" x="${tx}" y="${ty}" width="${tw}" height="${th}" rx="${fs * 0.3}"/>
        <text x="${tx + pad}" y="${ty + th * 0.72}" font-size="${fs}">${text}</text></g>`;
    });
    const names = [...new Set(v.boxes.map((b) => CONDITIONS[DET_KEYS[b.class]]?.name || b.display_name))].join(", ");
    return `<svg class="det-view" viewBox="0 0 ${v.w} ${v.h}" preserveAspectRatio="xMidYMid meet" role="img"
        aria-label="Your photo with outlined areas: ${names}" style="--sw:3px">
      <image href="${v.src}" width="${v.w}" height="${v.h}"/>${parts.join("")}</svg>`;
  }

  // Footer copy from the checkpoint's own test report.
  async function loadMeta() {
    try {
      const m = await (await fetch(`${API_BASE}/api/meta`)).json();
      const imgs = m.metrics?.images;
      if (m.task === "binary" && imgs) {
        const n = Object.values(imgs).reduce((a, b) => a + b, 0);
        $("#fairText").textContent = `The model learned from ${n.toLocaleString("en-SG")} photos from public datasets, not Singapore patients, so it can be wrong, especially in poor light. It can't see gum disease. We never suggest medicine.`;
      }
    } catch (e) { /* server not running: keep the default copy */ }
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

  // photoQuality() on a 32 px copy. Photos the browser can't decode (e.g. HEIC) go to the server as they are.
  async function quickCheck(url) {
    try {
      const img = await loadImage(url), S = 32;
      const c = document.createElement("canvas");
      c.width = c.height = S;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, S, S);
      return photoQuality(ctx.getImageData(0, 0, S, S).data, S);
    } catch (e) {
      return null;
    }
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
    const list = labelled();
    if (!list.length || $("#btnAnalyse").getAttribute("aria-disabled") === "true") return;
    $("#btnAnalyse").setAttribute("aria-disabled", "true");
    showLoading(list.find((m) => m.kind === "image") || list[0], list.some((m) => m.kind === "video"));
    try {
      const [res] = await Promise.all([predict(list, setProgress), wait(1200)]);
      if (res.unclear) showUnclear(res.media, res.why, res.title);
      else showResult(res);
    } finally {
      $("#btnAnalyse").setAttribute("aria-disabled", String(!state.media.length));
    }
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

  // media: a { kind, url } item, or a plain image URL (samples). Progress shows only for videos,
  // which can take a while frame by frame.
  function showLoading(media, withProgress = false) {
    showOnly("#resultLoading");
    const m = typeof media === "string" ? { kind: "image", url: media } : media;
    $("#scanMedia").innerHTML = m.kind === "video"
      ? `<video src="${m.url}" muted autoplay loop playsinline></video>` : `<img src="${m.url}" alt="">`;
    $("#scanProgress").hidden = !withProgress;
    setProgress(0, null);
    scrollToResult();
  }

  function setProgress(done, total) {
    const pct = total ? Math.min(100, (done / total) * 100) : 0;
    $("#scanBar").style.width = `${pct}%`;
    $("#scanBar").classList.toggle("is-indeterminate", !total);
    $("#scanText").textContent = !done ? "Reading frames…"
      : total ? `${Math.min(done, total).toLocaleString("en-SG")} of ${total.toLocaleString("en-SG")} checked`
      : `${done.toLocaleString("en-SG")} checked`;
  }

  function showUnclear(media, why, title = "We can't read this photo") {
    showOnly("#resultUnclear");
    $("#unclearPhoto").innerHTML = mediaHtml(media, "What you added");
    $("#unclearTitle").textContent = title;
    $("#unclearWhy").textContent = why;
    clearResult();
  }

  $("#btnRetake").addEventListener("click", () => {
    clearMedia();
    showOnly("#resultEmpty");
    if (consented()) (isTouch() ? $("#cameraInput") : $("#fileInput")).click();
  });

  // res: { key, what?, plural?, demo?, photoUrl, view?, kicker?, ring?, bars?, barsTitle?, lowConf?, notes? }.
  // Samples only carry `probs`; the ring and bars then default to the six-class layout.
  function showResult(res) {
    state.result = res;
    const c = CONDITIONS[res.key];
    const bars = res.bars || CLASS_ORDER.map((k) => ({ key: k, value: res.probs[k] }));
    const ringVal = res.ring ? res.ring.value : res.probs[res.key];
    const pct = Math.round(ringVal * 100);
    const what = res.demo ? "this sample" : res.what || "your photo";

    showOnly("#resultBody");

    const photo = $("#resultPhoto");
    photo.classList.toggle("is-art", !res.photoUrl && !res.view);
    photo.innerHTML = res.view ? detectionSvg(res.view)
      : res.photoUrl ? `<img src="${res.photoUrl}" alt="Your uploaded photo">` : Art.condition(res.key);
    $("#resultKicker").textContent = res.kicker || "Looks most like";
    $("#resultName").textContent = c.name;
    $("#resultAka").textContent = c.aka;
    $("#simTag").hidden = !res.demo;

    $("#ringTitle").textContent = res.ring?.title || "Likelihood";
    $("#ringText").textContent = res.ring?.text
      ? res.ring.text(what, res.plural ? "" : "s")
      : `How closely ${what} match${res.plural ? "" : "es"}. A dentist must confirm.`;
    const low = res.lowConf ?? (ringVal < LOW_CONFIDENCE ? "The model isn't sure. Try a brighter, closer shot." : null);
    $("#lowConf").hidden = !low;
    if (low) $("#lowConfText").textContent = low;
    const notes = res.notes || [];
    $("#photoNotes").hidden = !notes.length;
    $("#photoNotes").innerHTML = notes.map((n) => `<li><i data-lucide="${n.icon}"></i><span>${n.html}</span></li>`).join("");

    const ring = $("#ring");
    ring.style.setProperty("--rc", c.color);
    ring.setAttribute("aria-label", `${pct} percent: ${res.ring?.title || "likelihood"}`);
    animateNumber($("#ringVal"), pct, (v) => ring.style.setProperty("--p", v));

    const sorted = bars.slice().sort((a, b) => b.value - a.value);
    const healthy = res.key === "normal";
    $("#probs").hidden = healthy || !sorted.length;
    $("#probsFold").hidden = !healthy;
    $("#probsFold").open = false;
    $("#probs").setAttribute("aria-label", res.barsTitle || "Likelihood of each condition");
    $(healthy ? "#probsFoldBody" : "#probs").innerHTML =
      (res.barsTitle && !healthy ? `<p class="probs-title">${res.barsTitle}</p>` : "") +
      sorted.map(({ key: k, value }) => {
        const v = Math.round(value * 100);
        return `<div class="prob ${k === res.key ? "is-top" : ""}">
          <span class="cond-dot" style="--c:${CONDITIONS[k].color}" aria-hidden="true"></span>
          <span>${CONDITIONS[k].name}</span>
          <span class="prob-bar"><span style="--c:${CONDITIONS[k].color}" data-w="${v}"></span></span>
          <span class="prob-val">${v}%</span>
        </div>`;
      }).join("");
    requestAnimationFrame(() => $$(".prob-bar span").forEach((s) => (s.style.width = s.dataset.w + "%")));

    renderUnderstand(res.key);
    renderAction(res.key);
    showResultSections(true);
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
    $("#mapNote").innerHTML = `<i data-lucide="info"></i>Distances from ${a.label}.`;
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
        note.innerHTML = '<i data-lucide="locate-fixed"></i>Distances from your location (never stored).';
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
    showResult({ key, probs: makeProbs(key, 0.72 + seedRand() * 0.18, seedRand), demo: true, photoUrl: null });
  }

  const svgUrl = (svg) => "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);

  /* ---------------- Reset ---------------- */
  // The explainer and clinic sections (and their nav links) only exist once there is a result.
  function showResultSections(on) {
    $$(".needs-result, [data-needs-result]").forEach((el) => (el.hidden = !on));
  }

  function clearResult() {
    state.result = null;
    showResultSections(false);
  }

  $("#btnReset").addEventListener("click", () => {
    clearMedia();
    clearResult();
    showOnly("#resultEmpty");
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
  $$(".section-head, .card").forEach((el, i) => {
    el.classList.add("reveal");
    el.style.transitionDelay = `${(i % 3) * 70}ms`;
    revealObs.observe(el);
  });

  /* ---------------- Boot ---------------- */
  syncConsent();
  renderMedia();
  icons();
  loadMeta();
  const demo = new URLSearchParams(location.search).get("demo");
  if (demo && CONDITIONS[demo]) runDemo(demo, false);
})();
