/* Content for each screened condition, plus home-care products and clinic sample data.
   Classes match the Kaggle Oral Image Dataset (6 classes). Wording is plain-language and
   never states a diagnosis or names a medicine. */

window.CONDITIONS = {
  calculus: {
    name: "Tartar",
    aka: "Hardened plaque on the teeth",
    color: "var(--c-calculus)",
    what: "Plaque that has hardened into a yellow or brown crust, usually where teeth meet the gums. Brushing can't remove it.",
    causes: [
      { icon: "droplets", text: "Plaque is left on teeth after meals" },
      { icon: "clock", text: "Saliva hardens it within days" },
      { icon: "target", text: "Spots missed when brushing or flossing" },
      { icon: "cigarette", text: "Smoking makes it build faster" },
    ],
    flow: [
      { icon: "circle-dot", label: "Tartar builds at the gum line", time: "Now" },
      { icon: "flame", label: "Gums swell and bleed when brushing", time: "Weeks to months" },
      { icon: "arrow-down-to-line", label: "Gums pull away, bone starts to wear", time: "Months to years" },
      { icon: "circle-off", label: "Teeth loosen and may be lost", time: "Years" },
    ],
    urgency: { level: 2, icon: "calendar-check", title: "Book a cleaning in the next few weeks", detail: "Within 1–3 months is fine. Only a dentist's cleaning removes tartar." },
    procedure: "Cleaning (scaling)",
    products: [
      { id: "softbrush", why: "Reaches the gum line without hurting gums" },
      { id: "interdental", why: "Clears plaque before it hardens between teeth" },
      { id: "floss", why: "Gets the spots where tartar starts" },
      { id: "paste", why: "Fluoride keeps the enamel underneath strong" },
    ],
    clinicTypes: ["chas", "polyclinic"],
  },
  caries: {
    name: "Tooth decay",
    aka: "Also called a cavity",
    color: "var(--c-caries)",
    what: "A soft spot or hole in the tooth, made by acid from germs that feed on sugar.",
    causes: [
      { icon: "candy", text: "Sugary food and drinks through the day" },
      { icon: "bug", text: "Germs turn sugar into acid" },
      { icon: "shield-off", text: "Acid softens the tooth's hard outer layer" },
      { icon: "repeat", text: "Plaque not brushed away twice a day" },
    ],
    flow: [
      { icon: "circle-dot", label: "Chalky or brown spot on the tooth", time: "Now" },
      { icon: "snowflake", label: "A hole forms; cold and sweet sting", time: "Months" },
      { icon: "zap", label: "Reaches the nerve: pain and infection", time: "Months to a year" },
      { icon: "circle-off", label: "Root canal, or the tooth is removed", time: "Later" },
    ],
    urgency: { level: 2, icon: "clock-alert", title: "See a dentist within 2–4 weeks", detail: "Sooner if it hurts. Small holes are quicker and cheaper to fix." },
    procedure: "Filling",
    products: [
      { id: "paste", why: "Fluoride helps harden early soft spots" },
      { id: "softbrush", why: "Brush twice a day, 2 minutes each time" },
      { id: "floss", why: "Decay often starts between teeth" },
      { id: "gum", why: "Chew after meals to wash away acid" },
    ],
    clinicTypes: ["chas", "polyclinic"],
  },
  discolouration: {
    name: "Tooth staining",
    aka: "Yellow, brown or grey teeth",
    color: "var(--c-discolouration)",
    what: "Teeth that look yellow, brown or grey. Usually surface stains, but sometimes a change inside the tooth.",
    causes: [
      { icon: "coffee", text: "Coffee, tea and dark drinks" },
      { icon: "cigarette", text: "Smoking or chewing tobacco" },
      { icon: "hourglass", text: "The outer layer thins with age" },
      { icon: "hand", text: "A knock to a tooth can darken it" },
    ],
    flow: [
      { icon: "circle-dot", label: "Light surface stains", time: "Now" },
      { icon: "layers", label: "Stains deepen and build up", time: "Months" },
      { icon: "search", label: "One dark tooth may mean a hurt nerve", time: "Get that one checked", sev: 2 },
    ],
    urgency: { level: 1, icon: "calendar", title: "Mention it at your next check-up", detail: "Sooner if a single tooth turned dark after a knock." },
    procedure: "Check-up and cleaning",
    products: [
      { id: "straw", why: "Keeps coffee and tea off your front teeth" },
      { id: "softbrush", why: "Lifts surface stains without scratching" },
      { id: "paste", why: "Protects enamel while you brush" },
      { id: "floss", why: "Stops stains building up between teeth" },
    ],
    clinicTypes: ["chas", "polyclinic"],
  },
  ulcer: {
    name: "Mouth ulcer",
    aka: "A sore inside the mouth",
    color: "var(--c-ulcer)",
    what: "A small, painful, round sore inside the mouth, often white or yellow with a red edge.",
    causes: [
      { icon: "frown", text: "Biting your cheek or lip by accident" },
      { icon: "hand", text: "Brushing too hard, or a sharp tooth rubbing" },
      { icon: "brain", text: "Stress or feeling run down" },
      { icon: "flame", text: "Spicy, sour or crunchy food" },
    ],
    flow: [
      { icon: "circle-dot", label: "Sore and tender for a few days", time: "Days 1–4", sev: 1 },
      { icon: "trending-down", label: "Most heal by themselves", time: "1–2 weeks", sev: 0 },
      { icon: "triangle-alert", label: "Still there after 2 weeks? Get it checked", time: "After 2 weeks", sev: 2 },
    ],
    urgency: { level: 2, icon: "eye", title: "Watch it for up to 2 weeks", detail: "See a dentist or doctor if it doesn't heal, keeps coming back, or grows." },
    procedure: "Mouth check",
    products: [
      { id: "softbrush", why: "Extra-soft bristles won't rub the sore" },
      { id: "slsfree", why: "Gentler foam, less likely to sting" },
      { id: "rinse", why: "Alcohol-free, so it won't burn" },
    ],
    clinicTypes: ["polyclinic", "chas"],
  },
  hypodontia: {
    name: "Missing teeth",
    aka: "A gap where a tooth should be",
    color: "var(--c-hypodontia)",
    what: "A gap where a tooth is missing. Some people are born without it; more often in adults, a tooth was lost or removed.",
    causes: [
      { icon: "circle-off", text: "Lost to decay or gum disease" },
      { icon: "hand", text: "Knocked out in an accident" },
      { icon: "dna", text: "Never grew. This runs in families" },
    ],
    flow: [
      { icon: "circle-dot", label: "A gap where a tooth should be", time: "Now" },
      { icon: "move-horizontal", label: "Nearby teeth drift or tilt into it", time: "Months to years" },
      { icon: "utensils", label: "Bite and chewing feel uneven", time: "Later" },
    ],
    urgency: { level: 1, icon: "calendar", title: "Talk about it at a routine visit", detail: "A dentist can explain ways to fill the gap or keep teeth from drifting." },
    procedure: "Dentist consult",
    products: [
      { id: "interdental", why: "Cleans the sides of teeth next to the gap" },
      { id: "softbrush", why: "Keeps the gums around the gap healthy" },
      { id: "floss", why: "Stops decay in the teeth you still have" },
    ],
    clinicTypes: ["polyclinic", "specialist", "chas"],
  },
  normal: {
    name: "Looks healthy",
    aka: "Nothing we screen for was found",
    color: "var(--c-normal)",
    what: "Nothing in your photo looks like the conditions we screen for. Keep up the good habits.",
    causes: [
      { icon: "sparkles", text: "Brushing twice a day with fluoride paste" },
      { icon: "cup-soda", text: "Keeping sugary drinks low" },
      { icon: "calendar-check", text: "Check-ups catch problems early" },
    ],
    flow: [
      { icon: "check", label: "Healthy-looking teeth", time: "Now" },
      { icon: "calendar-check", label: "Check-up and cleaning", time: "Every 6–12 months" },
    ],
    urgency: { level: 1, icon: "smile", title: "No rush. Keep up check-ups", detail: "Photos can't show gum disease, so still see a dentist yearly." },
    procedure: "Routine check-up",
    headings: { causes: "What keeps teeth healthy", flow: "Keep it that way" },
    products: [
      { id: "softbrush", why: "Twice a day, 2 minutes each time" },
      { id: "paste", why: "Fluoride keeps teeth strong" },
      { id: "floss", why: "Once a day, between every tooth" },
    ],
    clinicTypes: ["chas", "polyclinic"],
  },
};

window.CLASS_ORDER = ["calculus", "caries", "discolouration", "ulcer", "hypodontia", "normal"];

window.URGENCY_COLORS = ["var(--u1)", "var(--u2)", "var(--u3)", "var(--u4)"];
window.URGENCY_LABELS = ["At check-up", "In weeks", "In days", "Today"];

window.PRODUCTS = {
  softbrush: { name: "Soft toothbrush", search: "soft bristle toothbrush" },
  paste: { name: "Fluoride toothpaste", search: "fluoride toothpaste 1450ppm" },
  slsfree: { name: "SLS-free toothpaste", search: "SLS free toothpaste" },
  floss: { name: "Dental floss", search: "dental floss" },
  interdental: { name: "Interdental brushes", search: "interdental brush" },
  rinse: { name: "Alcohol-free mouth rinse", search: "alcohol free mouthwash" },
  straw: { name: "Reusable straw", search: "reusable drinking straw" },
  gum: { name: "Sugar-free gum", search: "sugar free xylitol gum" },
};

/* Prototype sample data. Public clinics are real institutions with approximate coordinates;
   "CHAS clinic (sample)" entries are placeholders that the agent will replace with live
   results from the CHAS clinic locator. */
window.CLINICS = [
  { id: "amk", name: "Ang Mo Kio Polyclinic", type: "polyclinic", addr: "21 Ang Mo Kio Central 2", lat: 1.3743, lng: 103.8455 },
  { id: "tp", name: "Toa Payoh Polyclinic", type: "polyclinic", addr: "2003 Toa Payoh Lorong 8", lat: 1.3364, lng: 103.8566 },
  { id: "ys", name: "Yishun Polyclinic", type: "polyclinic", addr: "30A Yishun Central 1", lat: 1.4305, lng: 103.836 },
  { id: "hg", name: "Hougang Polyclinic", type: "polyclinic", addr: "89 Hougang Ave 4", lat: 1.3703, lng: 103.892 },
  { id: "ndcs", name: "National Dental Centre Singapore", type: "specialist", addr: "5 Second Hospital Ave", lat: 1.2795, lng: 103.835 },
  { id: "nucohs", name: "National University Centre for Oral Health", type: "specialist", addr: "9 Lower Kent Ridge Rd", lat: 1.2967, lng: 103.783 },
  { id: "chas1", name: "CHAS dental clinic · AMK Ave 10", type: "chas", addr: "Ang Mo Kio Ave 10", lat: 1.3689, lng: 103.8556, sample: true },
  { id: "chas2", name: "CHAS dental clinic · Bishan", type: "chas", addr: "Bishan St 22", lat: 1.3584, lng: 103.8469, sample: true },
  { id: "chas3", name: "CHAS dental clinic · Serangoon", type: "chas", addr: "Serangoon Central", lat: 1.3521, lng: 103.8732, sample: true },
];

/* Areas a user can pick instead of sharing GPS. Town centres, approximate. */
window.AREAS = [
  { label: "Ang Mo Kio", lat: 1.3691, lng: 103.8454 },
  { label: "Bedok", lat: 1.3236, lng: 103.9273 },
  { label: "Bishan", lat: 1.3526, lng: 103.8352 },
  { label: "Bukit Batok", lat: 1.349, lng: 103.7496 },
  { label: "Clementi", lat: 1.3162, lng: 103.7649 },
  { label: "Hougang", lat: 1.3612, lng: 103.8863 },
  { label: "Jurong East", lat: 1.3329, lng: 103.7436 },
  { label: "Jurong West", lat: 1.3404, lng: 103.709 },
  { label: "Pasir Ris", lat: 1.3721, lng: 103.9474 },
  { label: "Punggol", lat: 1.3984, lng: 103.9072 },
  { label: "Queenstown", lat: 1.2942, lng: 103.7861 },
  { label: "Sengkang", lat: 1.3868, lng: 103.8914 },
  { label: "Tampines", lat: 1.3496, lng: 103.9568 },
  { label: "Toa Payoh", lat: 1.3343, lng: 103.8563 },
  { label: "Woodlands", lat: 1.4382, lng: 103.789 },
  { label: "Yishun", lat: 1.4304, lng: 103.8354 },
];

window.CLINIC_TYPES = {
  chas: { label: "CHAS clinic", icon: "credit-card", tag: "CHAS subsidy" },
  polyclinic: { label: "Polyclinic", icon: "building-2", tag: "Subsidised · by appointment" },
  specialist: { label: "Specialist", icon: "hospital", tag: "Referral usually needed" },
};

/* Default location when the user does not share theirs: Nanyang Polytechnic, Ang Mo Kio. */
window.DEFAULT_LOCATION = { lat: 1.3801, lng: 103.849, label: "Ang Mo Kio" };
