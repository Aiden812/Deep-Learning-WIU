/* Hand-drawn SVG illustrations: teeth, conditions, photo angles and home-care products. */
(function () {
  let uid = 0;
  const id = (p) => `${p}${++uid}`;

  const TOOTH = "M22 30C18 10 40 2 60 12C80 2 102 10 98 30C96 52 92 64 88 74C85 96 82 126 72 130C62 133 64 100 60 92C56 100 58 133 48 130C38 126 35 96 32 74C28 64 24 52 22 30Z";
  const CROWN = "M14 44C8 16 32 2 52 10C62 14 78 14 88 10C108 2 132 16 126 44C123 72 118 98 106 122C98 138 42 138 34 122C22 98 17 72 14 44Z";

  function enamel(gid, tint) {
    const mid = tint || "#efe7dd";
    return `<radialGradient id="${gid}" cx=".34" cy=".26" r=".85">
      <stop offset="0" stop-color="#ffffff"/><stop offset=".45" stop-color="${mid}"/><stop offset="1" stop-color="#b9ab9c"/></radialGradient>`;
  }

  function gum(red) {
    const c = red ? "#d9727a" : "#e7a3a1";
    return `<path d="M-120 96 Q-40 80 0 92 Q20 78 34 86 Q60 70 86 86 Q100 78 124 92 Q170 82 240 96 L240 200 L-120 200Z" fill="${c}"/>
            <path d="M-120 112 Q60 92 240 112 L240 200 L-120 200Z" fill="${red ? "#c75e67" : "#dc9391"}"/>`;
  }

  function tooth(variant) {
    const g = id("en"), c = id("cl");
    const tint = variant === "discolouration" ? "#e2c48f" : null;
    let extra = "";
    if (variant === "normal") {
      extra = `<path d="M96 8l4 10 10 4-10 4-4 10-4-10-10-4 10-4z" fill="#4f9f7a"/>
               <path d="M14 46l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#4f9f7a" opacity=".7"/>`;
    } else if (variant === "calculus") {
      extra = `<g clip-path="url(#${c})">
        <path d="M26 66 Q34 58 42 66 Q50 56 60 66 Q70 56 78 66 Q86 58 94 66 L94 86 L26 86Z" fill="#d9a441"/>
        <circle cx="38" cy="72" r="4" fill="#b98228"/><circle cx="58" cy="76" r="5" fill="#b98228"/>
        <circle cx="78" cy="71" r="3.5" fill="#b98228"/><circle cx="68" cy="80" r="3" fill="#e9c47a"/>
        <circle cx="46" cy="80" r="3" fill="#e9c47a"/></g>`;
    } else if (variant === "caries") {
      extra = `<path d="M62 22 Q72 16 80 24 Q86 34 76 40 Q66 44 60 36 Q56 28 62 22Z" fill="#5a3322"/>
               <path d="M65 26 Q72 22 76 28 Q78 34 72 36 Q66 37 64 32Z" fill="#2c1810"/>
               <path d="M44 34 Q48 30 52 34 Q52 39 47 39Z" fill="#8a5a3c" opacity=".7"/>`;
    } else if (variant === "discolouration") {
      extra = `<g clip-path="url(#${c})" opacity=".55">
        <path d="M20 40 Q60 32 100 40 L100 48 Q60 40 20 48Z" fill="#a8743b"/>
        <path d="M24 58 Q60 50 96 58 L96 64 Q60 56 24 64Z" fill="#a8743b"/></g>`;
    }
    return `<svg viewBox="0 0 120 140" xmlns="http://www.w3.org/2000/svg">
      <defs>${enamel(g, tint)}<clipPath id="${c}"><path d="${TOOTH}"/></clipPath></defs>
      <path d="${TOOTH}" fill="url(#${g})"/>
      ${extra}
      ${gum(variant === "calculus")}
    </svg>`;
  }

  function ulcer() {
    const g1 = id("en"), g2 = id("en"), r = id("ur");
    return `<svg viewBox="0 0 120 140" xmlns="http://www.w3.org/2000/svg">
      <defs>${enamel(g1)}${enamel(g2)}
        <radialGradient id="${r}" cx=".5" cy=".5" r=".5"><stop offset=".6" stop-color="#d9576a" stop-opacity=".9"/><stop offset="1" stop-color="#d9576a" stop-opacity="0"/></radialGradient>
      </defs>
      <g transform="translate(2 -40) scale(.5)"><path d="${TOOTH}" fill="url(#${g1})" transform="rotate(180 60 70)"/></g>
      <g transform="translate(58 -40) scale(.5)"><path d="${TOOTH}" fill="url(#${g2})" transform="rotate(180 60 70)"/></g>
      <path d="M-120 34 Q-40 26 0 30 Q30 22 60 30 Q90 22 120 30 Q170 26 240 34 L240 200 L-120 200Z" fill="#e7a3a1"/>
      <path d="M-120 44 Q60 32 240 44" stroke="#f3c0bd" stroke-width="4" fill="none"/>
      <ellipse cx="60" cy="88" rx="42" ry="30" fill="url(#${r})"/>
      <ellipse cx="60" cy="88" rx="30" ry="19" fill="#f3e6c4"/>
      <ellipse cx="54" cy="83" rx="11" ry="5" fill="#fbf4e2"/>
    </svg>`;
  }

  function hypodontia() {
    const g1 = id("en"), g2 = id("en");
    return `<svg viewBox="0 0 120 140" xmlns="http://www.w3.org/2000/svg">
      <defs>${enamel(g1)}${enamel(g2)}</defs>
      <g transform="translate(-14 26) scale(.52)"><path d="${TOOTH}" fill="url(#${g1})"/></g>
      <g transform="translate(72 26) scale(.52)"><path d="${TOOTH}" fill="url(#${g2})"/></g>
      <g transform="translate(36 22) scale(.4)"><path d="${TOOTH}" fill="none" stroke="#6c84c4" stroke-width="6" stroke-dasharray="12 10" stroke-linecap="round"/></g>
      <path d="M-120 96 Q60 76 240 96 L240 200 L-120 200Z" fill="#e7a3a1"/>
      <path d="M-120 110 Q60 94 240 110 L240 200 L-120 200Z" fill="#dc9391"/>
    </svg>`;
  }

  function condition(key) {
    if (key === "ulcer") return ulcer();
    if (key === "hypodontia") return hypodontia();
    return tooth(key);
  }

  function heroTeeth() {
    const g = [id("en"), id("en"), id("en")], m = id("mt"), s = id("sh");
    const crowns = [
      { x: 0, y: 120, r: -7 },
      { x: 186, y: 64, r: 0 },
      { x: 372, y: 140, r: 6 },
    ].map((p, i) => `
      <g class="float"><g transform="translate(${p.x} ${p.y}) rotate(${p.r} 70 70) scale(1.3)">
        <rect x="64" y="128" width="12" height="600" rx="6" fill="url(#${m})"/>
        <path d="M54 126 L86 126 L80 150 L60 150Z" fill="url(#${m})"/>
        <path d="${CROWN}" fill="url(#${g[i]})"/>
        <path d="M30 40 C30 24 46 18 56 22" stroke="#ffffff" stroke-width="7" stroke-linecap="round" fill="none" opacity=".75"/>
        <ellipse cx="70" cy="136" rx="44" ry="6" fill="url(#${s})"/>
      </g></g>`).join("");
    return `<svg viewBox="0 0 560 640" preserveAspectRatio="xMidYMin slice" xmlns="http://www.w3.org/2000/svg">
      <defs>${g.map((x) => enamel(x)).join("")}
        <linearGradient id="${m}" x1="0" x2="1"><stop offset="0" stop-color="#4a4743"/><stop offset=".45" stop-color="#d8d3cd"/><stop offset="1" stop-color="#3a3835"/></linearGradient>
        <radialGradient id="${s}"><stop offset="0" stop-color="#000" stop-opacity=".35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
      </defs>${crowns}</svg>`;
  }

  function dropArt() {
    const g = id("en");
    return `<svg viewBox="0 0 160 120" xmlns="http://www.w3.org/2000/svg">
      <defs>${enamel(g)}</defs>
      <rect x="40" y="6" width="80" height="108" rx="16" fill="#171513"/>
      <rect x="47" y="16" width="66" height="88" rx="9" fill="#f6d8c3"/>
      <g transform="translate(58 26) scale(.37)"><path d="${TOOTH}" fill="url(#${g})"/></g>
      <path d="M52 22h10M52 22v10M108 22H98M108 22v10M52 98h10M52 98V88M108 98H98M108 98V88" stroke="#171513" stroke-width="3" stroke-linecap="round"/>
      <circle cx="132" cy="30" r="16" fill="#efbf9f"/><path d="M132 22v16M124 30h16" stroke="#171513" stroke-width="3.5" stroke-linecap="round"/>
    </svg>`;
  }

  function emptyArt() {
    const g = id("en");
    return `<svg viewBox="0 0 180 160" xmlns="http://www.w3.org/2000/svg">
      <defs>${enamel(g)}</defs>
      <circle cx="80" cy="76" r="62" fill="#ece2d9"/>
      <g transform="translate(40 20) scale(.66)"><path d="${TOOTH}" fill="url(#${g})"/></g>
      <circle cx="108" cy="92" r="28" fill="none" stroke="#171513" stroke-width="8"/>
      <path d="M128 112l26 26" stroke="#171513" stroke-width="12" stroke-linecap="round"/>
    </svg>`;
  }

  function angle(kind) {
    const ink = "#5d5650", t = "#ffffff";
    const teethRow = (y, n, x0, w) => Array.from({ length: n }, (_, i) =>
      `<rect x="${x0 + i * w}" y="${y}" width="${w - 2}" height="12" rx="3" fill="${t}" stroke="${ink}" stroke-width="1.5"/>`).join("");
    const arch = (flip) => {
      const pts = Array.from({ length: 9 }, (_, i) => {
        const a = Math.PI * (i / 8);
        const x = 40 + Math.cos(a) * 26, y = flip ? 30 + Math.sin(a) * 22 : 50 - Math.sin(a) * 22;
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5" fill="${t}" stroke="${ink}" stroke-width="1.5"/>`;
      });
      return pts.join("");
    };
    const inner = {
      front: `<ellipse cx="40" cy="40" rx="32" ry="20" fill="#e7a3a1"/>${teethRow(28, 6, 16, 8)}${teethRow(41, 6, 16, 8)}`,
      upper: `<path d="M10 52 Q40 -6 70 52" fill="#e7a3a1"/>${arch(false)}<path d="M40 64v10M34 68l6 6 6-6" stroke="${ink}" stroke-width="2" fill="none" stroke-linecap="round"/>`,
      lower: `<path d="M10 28 Q40 86 70 28" fill="#e7a3a1"/>${arch(true)}<path d="M40 16V6M34 12l6-6 6 6" stroke="${ink}" stroke-width="2" fill="none" stroke-linecap="round"/>`,
      left: `<rect x="20" y="22" width="48" height="36" rx="14" fill="#e7a3a1"/>${teethRow(28, 4, 26, 10)}${teethRow(41, 4, 26, 10)}<path d="M14 40H4M9 35l-5 5 5 5" stroke="${ink}" stroke-width="2" fill="none" stroke-linecap="round"/>`,
      right: `<rect x="12" y="22" width="48" height="36" rx="14" fill="#e7a3a1"/>${teethRow(28, 4, 16, 10)}${teethRow(41, 4, 16, 10)}<path d="M66 40h10M71 35l5 5-5 5" stroke="${ink}" stroke-width="2" fill="none" stroke-linecap="round"/>`,
    }[kind];
    return `<svg viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`;
  }

  function product(kind) {
    const k = "#171513", w = "#ffffff", a = "#efbf9f", b = "#4f9f7a";
    // Toothpaste tube, tilted, with a ribbon of paste coming out of the nozzle.
    function tube(stripe, leaf) {
      return `<g transform="rotate(-30 32 36)">
        <path d="M6 26 H40 L48 30 V40 L40 44 H6 Z" fill="${w}" stroke="${k}" stroke-width="2.5" stroke-linejoin="round"/>
        <path d="M10 26 V44" stroke="${k}" stroke-width="2"/>
        <rect x="48" y="31" width="7" height="8" rx="1.5" fill="${k}"/>
        <rect x="16" y="30" width="20" height="10" rx="3" fill="${stripe}"/>
        ${leaf ? `<path d="M22 39 C22 33 28 31 30 31 C30 37 26 39 22 39Z" fill="${b}"/>` : ""}
      </g>
      <path d="M50 14 C56 12 60 16 57 20 C54 23 49 20 52 17" stroke="${b}" stroke-width="4" fill="none" stroke-linecap="round"/>`;
    }
    const art = {
      toothbrush: `<rect x="8" y="40" width="44" height="9" rx="4.5" fill="${k}" transform="rotate(-30 30 44)"/>
        <g transform="rotate(-30 30 44)"><rect x="50" y="38" width="16" height="9" rx="3" fill="${k}"/>
        <rect x="51" y="30" width="3" height="9" rx="1.5" fill="${b}"/><rect x="55.5" y="29" width="3" height="10" rx="1.5" fill="${w}"/><rect x="60" y="30" width="3" height="9" rx="1.5" fill="${b}"/></g>`,
      paste: tube(b),
      slsfree: tube("#9cc9a8", true),
      floss: `<rect x="14" y="22" width="34" height="34" rx="10" fill="${w}" stroke="${k}" stroke-width="2.5"/>
        <path d="M14 34 H48" stroke="${k}" stroke-width="2.5"/><rect x="26" y="18" width="10" height="6" rx="2" fill="${k}"/>
        <path d="M31 18 C31 8 44 6 52 12 C58 17 56 28 50 32" stroke="${b}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <rect x="20" y="40" width="22" height="8" rx="3" fill="${a}"/>`,
      interdental: `<rect x="6" y="30" width="34" height="10" rx="5" fill="${k}"/><rect x="40" y="33" width="12" height="4" rx="2" fill="${k}"/>
        <g stroke="${b}" stroke-width="2" stroke-linecap="round"><path d="M52 29v12M55 28v14M58 29v12M61 31v8"/></g>`,
      rinse: `<rect x="20" y="16" width="24" height="10" rx="3" fill="${k}"/><path d="M16 26 H48 Q52 26 52 30 V56 Q52 60 48 60 H16 Q12 60 12 56 V30 Q12 26 16 26Z" fill="${w}" stroke="${k}" stroke-width="2.5"/>
        <rect x="12" y="36" width="40" height="14" fill="${a}"/>`,
      straw: `<path d="M16 26 H46 L42 60 H20Z" fill="${w}" stroke="${k}" stroke-width="2.5"/><path d="M18 36 H44 L42 60 H20Z" fill="#7a4b33"/>
        <path d="M36 44 L44 6 H54" stroke="${b}" stroke-width="4" fill="none" stroke-linecap="round"/>`,
      gum: `<rect x="8" y="30" width="24" height="15" rx="7.5" fill="${w}" stroke="${k}" stroke-width="2.5" transform="rotate(-18 20 37)"/>
        <rect x="28" y="38" width="24" height="15" rx="7.5" fill="${w}" stroke="${k}" stroke-width="2.5" transform="rotate(12 40 45)"/>
        <path d="M40 10 C52 8 58 16 56 26 C46 28 38 22 40 10Z" fill="${b}"/><path d="M42 12 L54 24" stroke="${w}" stroke-width="1.5"/>`,
      softbrush: `<rect x="8" y="40" width="44" height="9" rx="4.5" fill="${a}" stroke="${k}" stroke-width="2" transform="rotate(-30 30 44)"/>
        <g transform="rotate(-30 30 44)"><rect x="50" y="38" width="16" height="9" rx="3" fill="${k}"/>
        <g fill="${w}" stroke="${k}" stroke-width=".8"><circle cx="53" cy="33" r="3"/><circle cx="58" cy="32" r="3.2"/><circle cx="63" cy="33" r="3"/></g></g>`,
      scraper: `<path d="M14 54 Q30 10 50 54" stroke="${k}" stroke-width="7" fill="none" stroke-linecap="round"/><path d="M22 30 Q32 18 42 30" stroke="${b}" stroke-width="4" fill="none" stroke-linecap="round"/>`,
    }[kind] || "";
    return `<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">${art}</svg>`;
  }

  window.Art = { condition, heroTeeth, dropArt, emptyArt, angle, product };
})();
