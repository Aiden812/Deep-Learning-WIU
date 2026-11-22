const $ = (s) => document.querySelector(s);
// h("p", {className: "x"}, "text", node): children are appended as text nodes, so server strings are never parsed as HTML
const h = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids.flat().filter((k) => k != null && k !== false));
  return e;
};
const money = (n) => "S$" + n.toLocaleString("en-SG", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = (x) => Math.round(x * 100) + "%";
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Something went wrong. Please try again.");
  return j;
};
const STATUS = { flagged: "Flagged", borderline: "Borderline", clear: "Not flagged" };
const DEMOS = [["Tartar + discolouration", "calculus,tooth_discoloration"], ["Tooth decay", "caries"], ["Mouth ulcer", "mouth_ulcer"], ["Nothing flagged", ""]];

let META, FLAGGED = [], COORDS = null;

function step(name) {
  let past = true;
  for (const li of document.querySelectorAll(".steps li")) {
    const current = li.dataset.step === name;
    if (current) past = false;
    li.classList.toggle("done", past);
    current ? li.setAttribute("aria-current", "step") : li.removeAttribute("aria-current");
  }
}

function say(text) {
  $("#photo-msg").textContent = text;
  $("#photo-msg").hidden = !text;
}

async function screen(body) {
  say("");
  $("#go").disabled = true;
  try {
    renderResult(await api("/api/screen", { method: "POST", body }));
  } catch (e) {
    say(e.message);
  }
  $("#go").disabled = false;
}

function renderResult(r) {
  if (!r.quality.ok) return say(r.quality.issues.map((i) => i.message).join(" "));
  FLAGGED = r.flagged;
  const C = META.conditions, u = META.urgency[r.urgency], n = r.flagged.length;
  const box = $("#result");
  box.replaceChildren(
    r.simulated && h("p", { className: "msg" }, "Simulated result for demonstration. It did not come from the model."),
    ...r.quality.issues.map((i) => h("p", { className: "msg" }, i.message + " The result below may be less reliable.")),
    h("div", { className: "urgency" },
      h("strong", {}, n ? `${n} finding${n > 1 ? "s" : ""} flagged · ${u.label}` : "Nothing flagged"),
      n ? u.advice : C.normal.what + " " + C.normal.dentist),
    ...r.conditions.map((c) => {
      const info = C[c.id];
      return h("div", { className: "finding " + c.status },
        h("div", { className: "finding-head" }, h("b", {}, info.label), h("span", { className: "tag" }, STATUS[c.status]),
          h("span", { className: "pct" }, `score ${pct(c.probability)} · flags at ${pct(c.threshold)}`)),
        h("div", { className: "bar", role: "img", ariaLabel: `Score ${pct(c.probability)}, flagged from ${pct(c.threshold)}` },
          h("i", { style: `width:${pct(c.probability)}` }), h("s", { style: `left:${pct(c.threshold)}` })),
        c.status === "borderline" && h("p", { className: "note" }, `${info.short}. Just under the line: worth mentioning at your next dental visit.`),
        c.status === "flagged" && h("dl", {},
          h("dt", {}, "What it is"), h("dd", {}, info.what),
          h("dt", {}, "Why it happens"), h("dd", {}, info.why),
          h("dt", {}, "If it's left alone"), h("dd", {}, info.if_left),
          h("dt", {}, "What a dentist does"), h("dd", {}, info.dentist),
          h("dt", {}, "What you can do at home"), h("dd", {}, h("ul", {}, info.home.map((t) => h("li", {}, t))))));
    }),
    h("p", { className: "note" }, "This is a screening result, not a diagnosis. The score is the model's estimate that a condition is visible in this photo; the white line is where it starts flagging. Only a dentist examining you in person can confirm anything."));
  $("#step-result").hidden = $("#step-plan").hidden = false;
  $("#plan").replaceChildren();
  step("result");
  $("#step-result").focus();
}

function costRows(lines) {
  // data-label is shown beside each value on phones, where the table header is hidden
  const num = (label, kid) => { const td = h("td", { className: "num" }, kid); td.dataset.label = label; return td; };
  return lines.map((l) => h("tr", {},
    h("td", {}, h("b", {}, l.name), h("small", {}, l.plain)),
    num("Typical fee", `${money(l.fee_min)} to ${money(l.fee_max)}`),
    num("CHAS subsidy", l.subsidy_unknown ? "Ask clinic" : l.subsidy ? "− " + money(l.subsidy) : "None"),
    num("MediSave", l.medisave ? "− " + money(l.medisave) : "None"),
    num("You pay", h("b", {}, `${money(l.pay_min)} to ${money(l.pay_max)}`))));
}

function renderPlan(p) {
  const c = p.cost, t = c.likely_total;
  const head = h("tr", {}, ["Treatment", "Typical fee", "CHAS subsidy", "MediSave", "You pay"].map((x, i) => h("th", { className: i ? "num" : "" }, x)));
  $("#plan").replaceChildren(
    h("h3", {}, "Estimated cost"),
    c.likely.length ? h("p", { className: "total" }, `${money(t.pay_min)} to ${money(t.pay_max)}`) : null,
    h("p", { className: "note" }, c.likely.length ? "What you'd likely pay for the usual treatment, before GST. A non-binding estimate: the clinic's quote is what counts."
      : "There's no routine treatment fee for these findings. A consultation fee still applies and varies by clinic."),
    (c.likely.length || c.if_worse.length) ? h("div", { className: "tablewrap" }, h("table", {}, h("thead", {}, head), h("tbody", {},
      costRows(c.likely),
      c.if_worse.length ? h("tr", {}, h("th", { colSpan: 5 }, "Only if a dentist finds it has progressed")) : null,
      costRows(c.if_worse)))) : null,
    h("ul", { className: "note" }, c.notes.map((x) => h("li", {}, x))),
    h("p", { className: "note" }, "Sources: ", Object.values(c.sources).flatMap((s, i) => [i ? "; " : "", h("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title), ` (checked ${s.checked})`])),

    h("h3", {}, `Clinics near ${p.near}`),
    h("p", { className: "note" }, "Sample listings for the prototype, not real clinics. Subsidies only apply at CHAS clinics, so those rank higher if you hold a card."),
    p.clinics.length ? h("ul", { className: "list" }, p.clinics.map((k) => h("li", {},
      h("b", {}, k.name), h("small", {}, `${k.area} · ${k.distance_km} km · ${k.setting}`),
      h("div", { className: "chips" }, k.why.map((w) => h("span", {}, w))))))
      : h("p", {}, "No matching clinics found. Try another town."),

    h("h3", {}, "Home-care products to look for"),
    h("p", { className: "note" }, "Product types, not brands, and nothing medicated."),
    h("ul", { className: "list" }, p.products.map((x) => h("li", {}, h("b", {}, x.name), h("small", {}, x.why), h("small", {}, "Look for: " + x.look_for)))),

    h("div", { className: "urgency" }, h("strong", {}, "Before you go"), h("ul", {}, p.safety.advice.map((a) => h("li", {}, a)))),
    h("details", { className: "soft" }, h("summary", {}, `What the assistant did (${p.trace.length} steps)`),
      h("ol", { className: "trace" }, p.trace.map((s) => h("li", {}, h("b", {}, s.tool), ` ${JSON.stringify(s.args)} → ${s.result} (${s.ms} ms)`)))));
  step("plan");
}

async function plan() {
  const f = new FormData($("#profile"));
  const body = { conditions: FLAGGED, after_hours: f.has("after_hours"), area: f.get("area"), ...(COORDS || {}),
    profile: { residency: f.get("residency"), tier: f.get("tier") || "none", age: f.get("age") } };
  try {
    renderPlan(await api("/api/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
  } catch (e) {
    $("#plan").replaceChildren(h("p", { className: "msg" }, e.message));
  }
}

function renderMetrics(m) {
  if (!m) return;
  const rows = Object.entries(m.per_condition).map(([name, v]) => h("tr", {},
    h("td", {}, META.conditions[name.toLowerCase()].label),
    h("td", { className: "num" }, v.n),
    h("td", { className: "num" }, pct(v.recall)),
    h("td", { className: "num" }, v.precision == null ? "n/a" : pct(v.precision))));
  $("#metrics").replaceChildren(
    h("p", {}, `Tested once on ${m.n_test} photos the model had never seen in any form (copies of training photos were removed first). Conditions with fewer than 20 test photos are too few to judge.`),
    h("div", { className: "tablewrap" }, h("table", {}, h("thead", {}, h("tr", {},
      h("th", {}, "Condition"), h("th", { className: "num" }, "Test photos with it"), h("th", { className: "num" }, "Caught"), h("th", { className: "num" }, "Right when flagged"))),
      h("tbody", {}, rows))),
    h("p", { className: "note" }, `Overall ranking quality (macro average precision): ${m.macro_ap.toFixed(2)}, 95% interval ${m.macro_ap_ci[0].toFixed(2)} to ${m.macro_ap_ci[1].toFixed(2)}. The test set is small, so these numbers are rough.`
      + (m.healthy_flag_rate == null ? "" : ` On the ${m.n_normal} healthy mouths in that test, it wrongly flagged at least one condition ${pct(m.healthy_flag_rate)} of the time.`)));
}

META = await api("/api/meta");
if (META.mode === "demo") Object.assign($("#mode"), { hidden: false, textContent: "Demo mode: model not loaded", title: META.mode_reason });
$("#emergency-signs").append(...META.emergency_signs.map((s) => h("li", {}, s)));
$("#area").append(...META.areas.map((a) => h("option", { selected: a === "Toa Payoh" }, a)));
$("#demos").append(...DEMOS.map(([label, ids]) => h("button", { className: "btn", type: "button", onclick: () => { const b = new FormData(); b.set("demo", ids); screen(b); } }, label)));
renderMetrics(META.metrics);

$("#consent").onchange = (e) => {
  $("#file").disabled = !e.target.checked;
  $("#pick-label").setAttribute("aria-disabled", String(!e.target.checked));
};
$("#file").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  Object.assign($("#preview"), { src: URL.createObjectURL(file), hidden: false });
  $("#go").hidden = false;
  say("");
};
$("#go").onclick = () => { const b = new FormData(); b.set("file", $("#file").files[0]); screen(b); };
$("#profile").residency.onchange = (e) => { $("#profile").tier.disabled = e.target.value !== "citizen"; };   // CHAS is for citizens only
$("#profile").onsubmit = (e) => { e.preventDefault(); COORDS = null; plan(); };
$("#locate").onclick = () => navigator.geolocation.getCurrentPosition(
  (pos) => { COORDS = { lat: pos.coords.latitude, lng: pos.coords.longitude }; plan(); },
  () => $("#plan").replaceChildren(h("p", { className: "msg" }, "Couldn't get your location. Pick a town instead.")));

// Presenter deep link: ?demo=calculus,tooth_discoloration shows a simulated result; add &plan to run the cost and clinic step too
const q = new URLSearchParams(location.search);
if (q.has("demo")) {
  const b = new FormData();
  b.set("demo", q.get("demo"));
  await screen(b);
  if (q.has("plan")) await plan();
}
